---
layout: blog_detail
title: "INT4 Decoding GQA CUDA Optimizations for LLM Inference"
author: Sarunya Pumma, Jongsoo Park, Jianyu Huang, Amy Yang, Jaewon Lee, Daniel Haziza, Grigory Sizov, Jeremy Reizenstein, Jeff Johnson, Ying Zhang
---

#### An efficient decoding Grouped-Query Attention with low-precision KV cache

## Introduction

Generative AI has taken the world by storm with its ability to generate content like humans.  Many of these generative AI tools are powered by large language models (LLMs), like Meta [Llama](https://llama.meta.com/llama3/) models and OpenAI’s [ChatGPT](https://openai.com/gpt-4).  One of the main challenges of LLMs is supporting large “context lengths” (also known as “sequence lengths”).  The context length refers to the number of tokens that the model uses to understand the input context and generate responses.  Longer context lengths generally translate into higher precision and quality in the responses.  However, long context lengths are compute and memory intensive.  This is mainly due to the following reasons:



* The computational complexity of attention layers increases proportionally with the context length (the growth rate depends on the attention algorithm).  As a result, when using long context lengths, the attention layers can become a bottleneck, particularly during the prefill phase where attentions are compute bound.
* The KV cache size grows linearly with the context length, thus, putting higher pressure on the memory requirement and consequently slowing down the already memory-bound attention decoding.  Moreover, since the memory capacity is limited, the batch size reduces when the KV cache gets bigger, which generally results in a drop in throughput.

The computational complexity growth is difficult to solve compared to the other problem mentioned above.  One way to address the KV cache size growth problem is to use low precision KV cache.  From our experiments, group-wise INT4 quantization provides comparable results in terms of accuracy compared to BF16 KV cache during the decode phase in Meta Llama 2 inference.  However, we did not observe any latency improvement, despite reading 4x lesser data in attention decoding layers.  This means that the INT4 attention is 4x less efficient at utilizing precious HBM bandwidth than BF16 attention.

In this note, we discuss the CUDA optimizations that we applied to INT4 GQA (grouped-query attention – the attention layer that we use in the LLM inference phase) to improve its performance by up to **1.8x on the NVIDIA A100 GPU** and **1.9x on the NVIDIA H100 GPU**.



* The **optimized CUDA INT4 GQA** outperformed [INT4 Flash-Decoding GQA](https://pytorch.org/blog/flash-decoding/) (the best performing INT4 GQA that we used in the experiment mentioned above) by **1.4x-1.7x on A100** and **1.09x-1.3x on H100.**
* The **optimized CUDA INT4 GQA** performs better than **BF16 Flash-Decoding GQA** by **1.5x-1.7x on A100 and 1.4x-1.7x on H100.**


## Background


### GQA for LLM Inference 

[Grouped-Query Attention (GQA)](https://arxiv.org/abs/2305.13245) is a variant of multi-head attention (MHA) where each KV cache head is shared across a group of query heads.  Our LLM inference adopts GQA as an attention layer in both the prefill and decode phases in order to reduce the capacity requirement for the KV cache.  We use multiple GPUs in inference where the KV cache and query heads are distributed across GPUs.  Each GPU runs an attention layer with a single KV head and a group of Q heads.  Therefore, when viewed from a single GPU perspective, the GQA component can also be described as [MQA (Multi-Query Attention)](https://arxiv.org/abs/1911.02150).

The simplified workflow of decoding GQA is illustrated in Figure 1.  GQA takes three main inputs: input query (denoted `Q`), K cache (denoted `K`), and V cache (denoted `V`).  Our current GQA inference uses BF16 for `Q`, `K`, and `V`.



* `Q` is a 4D BF16 tensor of shape (`B`, `1`, <code class="language-plaintext highlighter-rouge">H<sub>Q</sub></code>, `D`)
* `K` is a 4D BF16 tensor of shape (`B`, <code class="language-plaintext highlighter-rouge">T<sub>max</sub></code>, <code class="language-plaintext highlighter-rouge">H<sub>KV</sub></code>, `D`)
* `V` is a 4D BF16 tensor of shape (`B`, <code class="language-plaintext highlighter-rouge">T<sub>max</sub></code>, <code class="language-plaintext highlighter-rouge">H<sub>KV</sub></code>, `D`)

_where_



* `B` is the batch size (the number of input prompts)
* <code class="language-plaintext highlighter-rouge">H<sub>Q</sub></code> is the number of query heads
* <code class="language-plaintext highlighter-rouge">H<sub>KV</sub></code> is the number of KV heads (<code class="language-plaintext highlighter-rouge">H<sub>Q</sub></code> must be divisible by <code class="language-plaintext highlighter-rouge">H<sub>KV</sub></code>)
* <code class="language-plaintext highlighter-rouge">T<sub>max</sub></code> is the maximum context length
* `D` is the head dimension (fixed to 128)

GQA is simply <code class="language-plaintext highlighter-rouge">bmm(softmax(bmm(Q, K<sup>T</sup>) / sqrt(D)), V)</code>.  This yields a single output tensor (denoted as `O`) which is a 4D BF16 tensor that has the same shape as `Q`.  Note that matrix multiplications are performed using BF16, however, accumulation and `softmax` are carried out in FP32.  We call this “BF16 GQA” as the KV cache is BF16.


![Figure 1: The simplified workflow of BF16 GQA for LLM inference](/assets/images/int4-decoding/fg1.png){:style="width:100%;display:block;max-width:500px;margin-left:auto;margin-right:auto;"}

**Figure 1** The simplified workflow of BF16 GQA for LLM inference


### INT4 GQA

To further reduce the size of the KV cache, we explore the possibility of using INT4 for KV cache instead of BF16.  We estimate the potential performance improvement by calculating the computational intensity (CI) of INT4 GQA and comparing it to that of BF16 GQA, as CI represents FLOPS per byte.  We compute the CI for <code class="language-plaintext highlighter-rouge">QK<sup>T</sup></code> and `PV` (as shown in Equation 1) as they take KV cache as an operand.  Note that we disregard the `Q` load as it is negligible compared to the KV cache.  We also ignore any intermediate data loads/stores that are not on global memory.  Thus, the CI only takes into account the computation FLOPS and KV cache loads.

  
![Equation 1](/assets/images/int4-decoding/eq.jpg){:style="width:100%;display:block;max-width:400px;margin-left:auto;margin-right:auto;"}

**Equation (1)**

 
Assuming that <code class="language-plaintext highlighter-rouge">H<sub>Q</sub></code> = 8 and <code class="language-plaintext highlighter-rouge">H<sub>KV</sub></code> = 1, CI for BF16 KV cache is 8 while CI for INT4 KV cache is 32.  The CIs indicate that both BF16 and INT4 GQAs are memory bound (the peak CIs for BF16 tensor cores for A100 and H100 are [312 TF / 2 TB/s = 141](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf) and [990 TF / 3.35 TB/s = 269](https://www.nvidia.com/en-us/data-center/h100/); note that these TF numbers are without sparsity).  Moreover, with INT4 KV cache, we should expect up to 4x performance improvement compared to BF16 GQA.

To enable INT4 KV cache support in GQA, we can dequantize the KV cache from INT4 to BF16 before passing it to the BF16 GQA operator.  However, since KV cache is typically large, copying it from/to global memory can be costly.  Moreover, decoding GQA is a memory bound operation (the memory unit is utilized much more heavily than the compute unit).  Figure 2 shows the NCU profile of the [FMHA CUTLASS BF16 GQA kernel in xFormers](https://github.com/facebookresearch/xformers/blob/9f6abadabdec17cd4b5c301632a44bf8216a7f35/xformers/csrc/attention/cuda/fmha/autogen/impl/cutlassF_bf16_aligned.cu#L33), which is one of the state of the art implementations of GQA.  From the figure, it is obvious that memory is a bottleneck.


![Figure 2: The NCU profile of the FMHA CUTLASS BF16 kernel in xFormers](/assets/images/int4-decoding/fg2.png){:style="width:100%"}

**Figure 2** The NCU profile of the [FMHA CUTLASS BF16 kernel in xFormers](https://github.com/facebookresearch/xformers/blob/9f6abadabdec17cd4b5c301632a44bf8216a7f35/xformers/csrc/attention/cuda/fmha/autogen/impl/cutlassF_bf16_aligned.cu#L33)

A more efficient alternative is to fuse INT4 dequantization with the GQA operation (shown in Figure 3).  In other words, having GQA read INT4 KV cache directly and perform the INT4 to BF16 conversion within the kernel.  This change can potentially reduce the amount of global memory reads required for the KV cache, which could lead to a decrease in latency.  We call this “INT4 GQA.”


![Figure 3: The workflow of fused INT4 GQA](/assets/images/int4-decoding/fg3.png){:style="width:100%;display:block;max-width:500px;margin-left:auto;margin-right:auto;"}

**Figure 3** The workflow of fused INT4 GQA

We list the state of the art implementations of GQA in the table below along with their features in Table 1.

**Table 1** State of the art GQA implementations

<table class="table table-bordered">
  <tr>
   <td><strong>Implementation</strong>
   </td>
   <td><strong>Denote</strong>
   </td>
   <td><strong>BF16 GQA</strong>
   </td>
   <td><strong>Fused INT4 GQA</strong>
   </td>
  </tr>
  <tr>
   <td><a href="https://pytorch.org/blog/flash-decoding/">Flash-Decoding</a> (Triton implementation)
   </td>
   <td>FD
   </td>
   <td>Yes
   </td>
   <td>Yes
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/Dao-AILab/flash-attention">Flash Attention (v2.3.3)</a>
   </td>
   <td>FA
   </td>
   <td>Yes
   </td>
   <td>No
   </td>
  </tr>
  <tr>
   <td>CUDA baseline
   </td>
   <td>CU
   </td>
   <td>Yes
   </td>
   <td>Yes
   </td>
  </tr>
</table>


All implementations, except for CU, support both split-K and non split-K.  CU only has the split-K implementation.  Only FA has a heuristic in the backend to determine whether to run the split-K or non split-K kernel.  For other implementations, users must explicitly choose which version to run.  In this note, we focus on long context lengths (in our experiments, we use a context length of 8192) and therefore opt for the split-K version wherever possible.

As the baseline, we measured the performance of the state of the art GQA implementations on NVIDIA A100 and H100 GPUs.  The latency (time in microseconds) and achieved bandwidth (GB/s) are reported in Table 2.  Note that we ran a range of split-Ks (from 2 to 128 splits) and reported the best performance for each implementation.  For all experiments, we use a context length of 8192.  For INT4 GQA, we used row-wise quantization (i.e., num quantized groups = 1).

**Table 2** Baseline GQA performance

On A100


<table class="table table-bordered">
  <tr>
   <td><strong>Time (us)</strong>
   </td>
   <td colspan="3" ><strong>BF16 GQA</strong>
   </td>
   <td colspan="3" ><strong>INT4 GQA</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Batch size</strong>
   </td>
   <td><strong>FD</strong>
   </td>
   <td><strong>FA</strong>
   </td>
   <td><strong>CU</strong>
   </td>
   <td><strong>FD</strong>
   </td>
   <td><strong>FA</strong>
   </td>
   <td><strong>CU</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>139
   </td>
   <td>133
   </td>
   <td>183
   </td>
   <td>137
   </td>
   <td>-
   </td>
   <td>143
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>245
   </td>
   <td>229
   </td>
   <td>335
   </td>
   <td>234
   </td>
   <td>-
   </td>
   <td>257
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>433
   </td>
   <td>555
   </td>
   <td>596
   </td>
   <td>432
   </td>
   <td>-
   </td>
   <td>455
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>826
   </td>
   <td>977
   </td>
   <td>1127
   </td>
   <td>815
   </td>
   <td>-
   </td>
   <td>866
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1607
   </td>
   <td>1670
   </td>
   <td>2194
   </td>
   <td>1581
   </td>
   <td>-
   </td>
   <td>1659
   </td>
  </tr>
</table>



<table class="table table-bordered">
  <tr>
   <td><strong>Effective Bandwidth (GB/s)</strong>
   </td>
   <td colspan="3" ><strong>BF16 GQA</strong>
   </td>
   <td colspan="3" ><strong>INT4 GQA</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Batch size</strong>
   </td>
   <td><strong>FD</strong>
   </td>
   <td><strong>FA</strong>
   </td>
   <td><strong>CU</strong>
   </td>
   <td><strong>FD</strong>
   </td>
   <td><strong>FA</strong>
   </td>
   <td><strong>CU</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>965
   </td>
   <td>1012
   </td>
   <td>736
   </td>
   <td>262
   </td>
   <td>-
   </td>
   <td>250
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>1097
   </td>
   <td>1175
   </td>
   <td>802
   </td>
   <td>305
   </td>
   <td>-
   </td>
   <td>278
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>1240
   </td>
   <td>968
   </td>
   <td>901
   </td>
   <td>331
   </td>
   <td>-
   </td>
   <td>314
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>1301
   </td>
   <td>1100
   </td>
   <td>954
   </td>
   <td>351
   </td>
   <td>-
   </td>
   <td>331
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1338
   </td>
   <td>1287
   </td>
   <td>980
   </td>
   <td>362
   </td>
   <td>-
   </td>
   <td>345
   </td>
  </tr>
</table>


On H100


<table class="table table-bordered">
  <tr>
   <td><strong>Time (us)</strong>
   </td>
   <td colspan="3" ><strong>BF16 GQA</strong>
   </td>
   <td colspan="3" ><strong>INT4 GQA</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Batch size</strong>
   </td>
   <td><strong>FD</strong>
   </td>
   <td><strong>FA</strong>
   </td>
   <td><strong>CU</strong>
   </td>
   <td><strong>FD</strong>
   </td>
   <td><strong>FA</strong>
   </td>
   <td><strong>CU</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>91
   </td>
   <td>90
   </td>
   <td>114
   </td>
   <td>70
   </td>
   <td>-
   </td>
   <td>96
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>148
   </td>
   <td>146
   </td>
   <td>200
   </td>
   <td>113
   </td>
   <td>-
   </td>
   <td>162
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>271
   </td>
   <td>298
   </td>
   <td>361
   </td>
   <td>205
   </td>
   <td>-
   </td>
   <td>294
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>515
   </td>
   <td>499
   </td>
   <td>658
   </td>
   <td>389
   </td>
   <td>-
   </td>
   <td>558
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1000
   </td>
   <td>1011
   </td>
   <td>1260
   </td>
   <td>756
   </td>
   <td>-
   </td>
   <td>1066
   </td>
  </tr>
</table>



<table class="table table-bordered">
  <tr>
   <td><strong>Effective Bandwidth (GB/s)</strong>
   </td>
   <td colspan="3" ><strong>BF16 GQA</strong>
   </td>
   <td colspan="3" ><strong>INT4 GQA</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Batch size</strong>
   </td>
   <td><strong>FD</strong>
   </td>
   <td><strong>FA</strong>
   </td>
   <td><strong>CU</strong>
   </td>
   <td><strong>FD</strong>
   </td>
   <td><strong>FA</strong>
   </td>
   <td><strong>CU</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>1481
   </td>
   <td>1496
   </td>
   <td>1178
   </td>
   <td>511
   </td>
   <td>-
   </td>
   <td>371
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>1815
   </td>
   <td>1840
   </td>
   <td>1345
   </td>
   <td>631
   </td>
   <td>-
   </td>
   <td>443
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>1982
   </td>
   <td>1802
   </td>
   <td>1487
   </td>
   <td>699
   </td>
   <td>-
   </td>
   <td>487
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>2087
   </td>
   <td>2156
   </td>
   <td>1634
   </td>
   <td>736
   </td>
   <td>-
   </td>
   <td>513
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>2150
   </td>
   <td>2127
   </td>
   <td>1706
   </td>
   <td>757
   </td>
   <td>-
   </td>
   <td>537
   </td>
  </tr>
</table>


First, let’s discuss the BF16 GQA performance: CU ranks last in terms of performance among all implementations.  FD and FA have comparable performance.  When the batch size is less than or equal to 64, FA utilizes the split-K kernel and performs slightly better than FD.  However, when the batch size is greater than 64, FD performs better.

The same trend holds true for INT4 GQAs. However, we did not measure the performance of FA as it does not support INT4 KV cache. FD outperforms CU for all cases.

When comparing the latencies of FD between BF16 and INT4 GQAs, we find that they are almost identical.  This suggests that _INT4 GQA is highly inefficient_, which can be further confirmed by the significantly lower achievable bandwidth for INT4 GQA compared to BF16 GQA.  The same trend is also true when looking at the performance of CU.


### CUDA with Tensor Cores INT4 GQA Implementation

In this section, we briefly describe our baseline implementation which is CUDA with tensor cores INT4 GQA (CU).  Each thread block processes only one KV head and a group of query heads from one input prompt.  Therefore, each thread block performs <code class="language-plaintext highlighter-rouge">mm(softmax(mm(Q, K<sup>T</sup>) / sqrt(D)), V)</code>; notice that `mm` is being performed not `bmm`.  Moreover, since this is a split-K implementation, tokens in the KV cache are split among different thread blocks.  Note that each thread block contains 4 warps (each warp contains 32 threads for NVIDIA A100 and H100 GPUs).  Work in each thread block is split among warps.  Within each warp, we use the [WMMA](https://bruce-lee-ly.medium.com/nvidia-tensor-core-introduction-to-wmma-api-programming-21bcfee4ec45) API to compute matrix multiplication on tensor cores.  Figure 4 demonstrates the work partitioning in CU.


![Figure 4: CU work partitioning](/assets/images/int4-decoding/fg4.jpg){:style="width:100%"}


**Figure 4** CU work partitioning


## Optimizing CUDA with Tensor Cores Kernel of INT4 GQA

In this note, we discuss the optimizations that we have applied to the CUDA with tensor cores implementation of INT4 GQA (CU).  The ideal goal is to improve the INT4 GQA performance by 4 times based on the CI analysis in the previous section.  Note that the query size is negligible compared to the KV cache size when the context length is long.

In our analysis, we used the [NVIDIA Nsight Compute (NCU)](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) as the main profiler.  Our general bottleneck elimination approach is to minimize the stall cycles.  We applied 10 optimizations to INT4 GQA, three of which are specific for NVIDIA A100/H100 GPUs.  These optimizations are well known CUDA optimization techniques which can be generalized to many applications.

It is worth noting that the reason that we choose to optimize the CUDA implementation rather than the Flash-Decoding implementation (FD) (which is Triton based) is because with CUDA, we have a better control of how the low-level instructions are being generated.  Many optimization techniques that we apply such as, operating on tensor core fragments directly (Optimizations 7-9), cannot be done through Triton since it does not expose low-level details to developers.  However, these optimizations can be integrated into the compiler-based solution to make the optimizations available to broader operators, which is indeed a part of our future plan.


### Optimization 1: Unroll `K` Loads

**Problem Analysis:**

The NCU profile shows that during `K` loading, there are only 2 global loads followed by _memory stalls_ at `dequantize_permuted_int4`.  The memory stalls are the long scoreboard stalls which indicates the waits for global memory access.  This suggests that the kernel does not issue sufficient memory loads

to hide the global load latency.  The kernel issues data loading, and then waits to consume the data immediately causing the global load latency to be exposed.  The stalls are shown in Figure 5.


![Figure 5: K loading before unrolling](/assets/images/int4-decoding/fg5.png){:style="width:100%"}

**Figure 5** K loading before unrolling (the numbers that the arrows point to are stall cycles caused by global memory wait)

**Solution:**

In the baseline implementation, we use `uint32_t` to load 8 INT4 `K` values in a single load and we perform 2 `uint32_t` loads in each iteration, which is 16 INT4 K values.  To allow for a better global load latency hiding, we issue 8 `uint32_t` loads instead of two before consuming the `K` values in `dequantize_permuted_int4`.  This allows the compiler to unroll the loads as well as reorder the instructions to hide the global load latency better.  Figure 6 shows the NCU profile of `K` loading after unrolling.  Comparing Figure 5 and Figure 6, we effectively reduce the stall cycles by unrolling the `K` loads.


![Figure 6: K loading after unrolling](/assets/images/int4-decoding/fg6.png){:style="width:100%"}

**Figure 6** K loading after unrolling (the numbers that the arrows point to are stall cycles caused by global memory wait)

**Results:**

**Table 3** Performance of Optimization 1 for INT4 GQA (row-wise quantization)


<table class="table table-bordered">
  <tr>
   <td rowspan="3" ><strong>Batch size</strong>
   </td>
   <td colspan="3" ><strong>Time (us)</strong>
   </td>
   <td colspan="3" ><strong>Bandwidth (GB/s)</strong>
   </td>
   <td colspan="2" ><strong>Speed up</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CU</strong>
   </td>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CU</strong>
   </td>
   <td rowspan="2" ><strong>vs FD</strong>
   </td>
   <td rowspan="2" ><strong>vs CU baseline</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Baseline</strong>
   </td>
   <td><strong>Opt 1</strong>
   </td>
   <td><strong>Baseline</strong>
   </td>
   <td><strong>Opt 1</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>137
   </td>
   <td>143
   </td>
   <td>134
   </td>
   <td>262
   </td>
   <td>250
   </td>
   <td>267
   </td>
   <td><strong>1.02</strong>
   </td>
   <td><strong>1.07</strong>
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>234
   </td>
   <td>257
   </td>
   <td>237
   </td>
   <td>305
   </td>
   <td>278
   </td>
   <td>302
   </td>
   <td><strong>0.99</strong>
   </td>
   <td><strong>1.09</strong>
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>432
   </td>
   <td>455
   </td>
   <td>422
   </td>
   <td>331
   </td>
   <td>314
   </td>
   <td>339
   </td>
   <td><strong>1.02</strong>
   </td>
   <td><strong>1.08</strong>
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>815
   </td>
   <td>866
   </td>
   <td>806
   </td>
   <td>351
   </td>
   <td>331
   </td>
   <td>355
   </td>
   <td><strong>1.01</strong>
   </td>
   <td><strong>1.07</strong>
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1581
   </td>
   <td>1659
   </td>
   <td>1550
   </td>
   <td>362
   </td>
   <td>345
   </td>
   <td>369
   </td>
   <td><strong>1.02</strong>
   </td>
   <td><strong>1.07</strong>
   </td>
  </tr>
</table>



### Optimization 2: Improve `P` Type Casting (FP32->BF16)

**Problem Analysis:**

Since the product of <code class="language-plaintext highlighter-rouge">softmax(bmm(Q, K<sup>T</sup>) / sqrt(D))</code> is FP32 (denoted as `P` in Figure 3), the kernel has to convert `P` from FP32 to BF16 before feeding it to the next `bmm` computation.  The kernel performs the FP32 to BF16 conversion of `P` by copying the FP32 data from one location in shared memory to another location in shared memory.  This causes stalls during the shared memory access (shown in Figure 7) which might be caused by (1) the shared memory indirection; and (2) the shared memory bank conflict since each thread accesses an 16-bit element (because of this, two threads can access the same memory bank simultaneously).


![Figure 7: P type casting before Optimization 2](/assets/images/int4-decoding/fg7.png){:style="width:100%"}


**Figure 7** `P` type casting before Optimization 2 (the number that the arrow points to is stall cycles caused by shared memory wait)

**Solution:**

We use all threads in the thread block to do in-place type conversion.  Each thread operates on two consecutive elements in order to avoid the shared memory bank conflict when storing BF16.  All threads work on the same head (`h`) at the same time to guarantee correctness of the conversion.  The in-place conversion steps are as follows:



1. Each thread loads 2 FP32 token elements from the same head from the shared memory into registers
2. Call `__syncthreads()` to make sure that every thread finishes reading the data
3. Each thread converts its data to 2 BF16 token elements and then stores the results to the same shared memory

Some optimizations that we apply to the implementation:



* Use vector types (especially `nv_bfloat2`)
* Unroll data loading/storing, i.e., performing multiple loads before calling `__syncthreads()` and performing multiple stores after `__syncthreads()`

After this optimization, long stalls are not observed during `P` type casting as shown in Figure 8.

![Figure 8: P type casting after Optimization 2](/assets/images/int4-decoding/fg8.png){:style="width:100%"}

**Figure 8** `P` type casting after Optimization 2 (the numbers that the arrow points to are stall cycles caused by shared memory wait)

**Culprits:**

Since we unroll data loading/storing by using registers as an intermediate storage, the number of registers per thread increases resulting in reduced occupancy.

**Results:**

**Table 4** Performance of Optimization 2 for INT4 GQA (row-wise quantization)


<table class="table table-bordered">
  <tr>
   <td rowspan="3" ><strong>Batch size</strong>
   </td>
   <td colspan="3" ><strong>Time (us)</strong>
   </td>
   <td colspan="3" ><strong>Bandwidth (GB/s)</strong>
   </td>
   <td colspan="2" ><strong>Speed up</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CU</strong>
   </td>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CU</strong>
   </td>
   <td rowspan="2" ><strong>vs FD</strong>
   </td>
   <td rowspan="2" ><strong>vs CU baseline</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Baseline</strong>
   </td>
   <td><strong>Opt 2</strong>
   </td>
   <td><strong>Baseline</strong>
   </td>
   <td><strong>Opt 2</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>137
   </td>
   <td>143
   </td>
   <td>126
   </td>
   <td>262
   </td>
   <td>250
   </td>
   <td>285
   </td>
   <td><strong>1.09</strong>
   </td>
   <td><strong>1.14</strong>
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>234
   </td>
   <td>257
   </td>
   <td>221
   </td>
   <td>305
   </td>
   <td>278
   </td>
   <td>324
   </td>
   <td><strong>1.06</strong>
   </td>
   <td><strong>1.16</strong>
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>432
   </td>
   <td>455
   </td>
   <td>395
   </td>
   <td>331
   </td>
   <td>314
   </td>
   <td>362
   </td>
   <td><strong>1.09</strong>
   </td>
   <td><strong>1.15</strong>
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>815
   </td>
   <td>866
   </td>
   <td>749
   </td>
   <td>351
   </td>
   <td>331
   </td>
   <td>382
   </td>
   <td><strong>1.09</strong>
   </td>
   <td><strong>1.16</strong>
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1581
   </td>
   <td>1659
   </td>
   <td>1435
   </td>
   <td>362
   </td>
   <td>345
   </td>
   <td>399
   </td>
   <td><strong>1.10</strong>
   </td>
   <td><strong>1.16</strong>
   </td>
  </tr>
</table>



### Optimization 3: Remove Local Memory Usage for max <code class="language-plaintext highlighter-rouge">QK<sup>T</sup></code> computation 

**Problem Analysis:**

During the softmax computation, the kernel has to compute max <code class="language-plaintext highlighter-rouge">QK<sup>T</sup></code> for each head. It uses a temporary "thread-local" storage for storing per-thread max <code class="language-plaintext highlighter-rouge">QK<sup>T</sup></code> results (one float value for each head).  Depending on the compiler, the thread-local storage can be allocated on registers (on chip) or the local memory (off chip == global memory).  Unfortunately, in the baseline, the thread-local storage resides in the local memory which is much slower than the registers (shown in Figure 9).  We suspect that this is because the compiler cannot determine the indices of thread-local storage at compile time (since the number of heads (`H`) in the kernel is a runtime variable).  Accessing local memory as if accessing registers can hurt the performance of the kernel.


![Figure 9: Local memory access during max QKT computation](/assets/images/int4-decoding/fg9.png){:style="width:100%"}

**Figure 9** Local memory access during max <code class="language-plaintext highlighter-rouge">QK<sup>T</sup></code> computation

**Solution:**

We realize that we do not need `H` (number of heads) floats as temporary storage per thread since each thread can compute max <code class="language-plaintext highlighter-rouge">QK<sup>T</sup></code> for only one head instead of all the heads.  Thus, we only need one float per thread, which can be easily stored in a register.  To accumulate the max results among warps, we use shared memory.  This optimization eliminates the local memory usage during max <code class="language-plaintext highlighter-rouge">QK<sup>T</sup></code> computation.

**Results:**

**Table 5** Performance of Optimization 3 for INT4 GQA (row-wise quantization)


<table class="table table-bordered">
  <tr>
   <td rowspan="3" ><strong>Batch size</strong>
   </td>
   <td colspan="3" ><strong>Time (us)</strong>
   </td>
   <td colspan="3" ><strong>Bandwidth (GB/s)</strong>
   </td>
   <td colspan="2" ><strong>Speed up</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CU</strong>
   </td>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CU</strong>
   </td>
   <td rowspan="2" ><strong>vs FD</strong>
   </td>
   <td rowspan="2" ><strong>vs CU baseline</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Baseline</strong>
   </td>
   <td><strong>Opt 3</strong>
   </td>
   <td><strong>Baseline</strong>
   </td>
   <td><strong>Opt 3</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>137
   </td>
   <td>143
   </td>
   <td>119
   </td>
   <td>262
   </td>
   <td>250
   </td>
   <td>300
   </td>
   <td><strong>1.14</strong>
   </td>
   <td><strong>1.20</strong>
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>234
   </td>
   <td>257
   </td>
   <td>206
   </td>
   <td>305
   </td>
   <td>278
   </td>
   <td>348
   </td>
   <td><strong>1.14</strong>
   </td>
   <td><strong>1.25</strong>
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>432
   </td>
   <td>455
   </td>
   <td>368
   </td>
   <td>331
   </td>
   <td>314
   </td>
   <td>389
   </td>
   <td><strong>1.17</strong>
   </td>
   <td><strong>1.24</strong>
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>815
   </td>
   <td>866
   </td>
   <td>696
   </td>
   <td>351
   </td>
   <td>331
   </td>
   <td>411
   </td>
   <td><strong>1.17</strong>
   </td>
   <td><strong>1.24</strong>
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1581
   </td>
   <td>1659
   </td>
   <td>1338
   </td>
   <td>362
   </td>
   <td>345
   </td>
   <td>428
   </td>
   <td><strong>1.18</strong>
   </td>
   <td><strong>1.24</strong>
   </td>
  </tr>
</table>



### Optimization 4: Remove local memory usage for row sum

**Problem Analysis:**

Similar to[ ](https://www.internalfb.com/diff/D50183201)Optimization 3, the local memory usage problem is also observed during the row sum computation in the `softmax` computation.  Since local memory is off chip, accessing it as if accessing registers can hurt the performance of the kernel.

**Solution**:

We apply the same solution as the max <code class="language-plaintext highlighter-rouge">QK<sup>T</sup></code> computation for the row sum computation.  That is to have each thread compute a row sum of only one head, which requires only one float per thread.  This eliminates the need for local memory.

**Results:**

**Table 6** Performance of Optimization 4 for INT4 GQA (row-wise quantization)


<table class="table table-bordered">
  <tr>
   <td rowspan="3" ><strong>Batch size</strong>
   </td>
   <td colspan="3" ><strong>Time (us)</strong>
   </td>
   <td colspan="3" ><strong>Bandwidth (GB/s)</strong>
   </td>
   <td colspan="2" ><strong>Speed up</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CU</strong>
   </td>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CU</strong>
   </td>
   <td rowspan="2" ><strong>vs FD</strong>
   </td>
   <td rowspan="2" ><strong>vs CU baseline</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Baseline</strong>
   </td>
   <td><strong>Opt 4</strong>
   </td>
   <td><strong>Baseline</strong>
   </td>
   <td><strong>Opt 4</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>137
   </td>
   <td>143
   </td>
   <td>118
   </td>
   <td>262
   </td>
   <td>250
   </td>
   <td>302
   </td>
   <td><strong>1.15</strong>
   </td>
   <td><strong>1.21</strong>
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>234
   </td>
   <td>257
   </td>
   <td>204
   </td>
   <td>305
   </td>
   <td>278
   </td>
   <td>351
   </td>
   <td><strong>1.15</strong>
   </td>
   <td><strong>1.26</strong>
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>432
   </td>
   <td>455
   </td>
   <td>364
   </td>
   <td>331
   </td>
   <td>314
   </td>
   <td>393
   </td>
   <td><strong>1.19</strong>
   </td>
   <td><strong>1.25</strong>
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>815
   </td>
   <td>866
   </td>
   <td>688
   </td>
   <td>351
   </td>
   <td>331
   </td>
   <td>416
   </td>
   <td><strong>1.18</strong>
   </td>
   <td><strong>1.26</strong>
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1581
   </td>
   <td>1659
   </td>
   <td>1328
   </td>
   <td>362
   </td>
   <td>345
   </td>
   <td>431
   </td>
   <td><strong>1.19</strong>
   </td>
   <td><strong>1.25</strong>
   </td>
  </tr>
</table>



### Optimization 5: Add prefetch for `V` load

**Problem Analysis:**

The same issue as `K` loading is observed when loading `V`.  That is, the kernel issues data loading, and then waits to consume the data immediately causing the global load latency to be exposed.  However, when using the unrolling technique mentioned above, the compiler allocates the temporary buffer on local memory instead of registers causing a large slow down.

**Solution:**

We adopt the data prefetching technique for `V` loading.  We load the next iteration `V` values immediately after the current iteration values are consumed.  This allows the data loading to be overlapped with the `PK` computation resulting in better kernel performance.

**Results:**

**Table 7** Performance of Optimization 5 for INT4 GQA (row-wise quantization)


<table class="table table-bordered">
  <tr>
   <td rowspan="3" ><strong>Batch size</strong>
   </td>
   <td colspan="3" ><strong>Time (us)</strong>
   </td>
   <td colspan="3" ><strong>Bandwidth (GB/s)</strong>
   </td>
   <td colspan="2" ><strong>Speed up</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CU</strong>
   </td>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CU</strong>
   </td>
   <td rowspan="2" ><strong>vs FD</strong>
   </td>
   <td rowspan="2" ><strong>vs CU baseline</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Baseline</strong>
   </td>
   <td><strong>Opt 5</strong>
   </td>
   <td><strong>Baseline</strong>
   </td>
   <td><strong>Opt 5</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>137
   </td>
   <td>143
   </td>
   <td>109
   </td>
   <td>262
   </td>
   <td>250
   </td>
   <td>327
   </td>
   <td><strong>1.25</strong>
   </td>
   <td><strong>1.31</strong>
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>234
   </td>
   <td>257
   </td>
   <td>194
   </td>
   <td>305
   </td>
   <td>278
   </td>
   <td>370
   </td>
   <td><strong>1.21</strong>
   </td>
   <td><strong>1.33</strong>
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>432
   </td>
   <td>455
   </td>
   <td>345
   </td>
   <td>331
   </td>
   <td>314
   </td>
   <td>414
   </td>
   <td><strong>1.25</strong>
   </td>
   <td><strong>1.32</strong>
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>815
   </td>
   <td>866
   </td>
   <td>649
   </td>
   <td>351
   </td>
   <td>331
   </td>
   <td>441
   </td>
   <td><strong>1.26</strong>
   </td>
   <td><strong>1.33</strong>
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1581
   </td>
   <td>1659
   </td>
   <td>1244
   </td>
   <td>362
   </td>
   <td>345
   </td>
   <td>460
   </td>
   <td><strong>1.27</strong>
   </td>
   <td><strong>1.33</strong>
   </td>
  </tr>
</table>



### Optimization 6: Add Group-Wise INT4 (Groups = 4) with Vector Load

**Problem Analysis:**

Prior to this optimization, CU only supported row-wise INT4 quantization.  That is, every column in each row shares the same scales.  The scales of each row are stored in the first 4 bytes of each row as shown in Figure 10.  In the kernel, each thread loads only one row at a time.  Since each row contains 68 bytes (4 bytes for scales and 64 bytes for data), it cannot guarantee that every row aligns with a size of any vector type.  Thus, vector loads cannot be used for loading the KV cache.


![Figure 10: The layout of each row of INT4 KV cache with row-wise quantization](/assets/images/int4-decoding/fg10.jpg){:style="width:100%;display:block;max-width:500px;margin-left:auto;margin-right:auto;"}


**Figure 10** The layout of each row of INT4 KV cache with row-wise quantization

**Solution:**

We have implemented support for group-wise INT4 quantization with num groups = 4.  In this case, columns in each row in the KV cache tensor are divided into 4 equal groups.  Columns within the same group share the same scales for quantization/dequantization.  The data layout for INT4 KV cache is shown in Figure 11.   The scales for all groups are serialized and stored at the beginning of each row.  The INT4 data is also serialized and laid out next to the scales.

Because the number of bytes in each row now becomes 80 bytes, we can use a vector type, i.e., `uint2` in our case, to load data.  (We **do not** use `uint4` since each thread loads only 16 INT4s at a time due to the tensor core fragment size.)  Vector load is generally better than scalar load since it does not cause extra byte loads.


![Figure 11: The layout of each row of INT4 KV cache with row-wise quantization](/assets/images/int4-decoding/fg11.jpg){:style="width:100%;display:block;max-width:500px;margin-left:auto;margin-right:auto;"}

**Figure 11** The layout of each row of INT4 KV cache with row-wise quantization

**Results:**

**Table 8** Performance of Optimization 6 for INT4 GQA (row-wise quantization)


<table class="table table-bordered">
  <tr>
   <td rowspan="3" ><strong>Batch size</strong>
   </td>
   <td colspan="3" ><strong>Time (us)</strong>
   </td>
   <td colspan="3" ><strong>Bandwidth (GB/s)</strong>
   </td>
   <td colspan="2" ><strong>Speed up</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CU</strong>
   </td>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CU</strong>
   </td>
   <td rowspan="2" ><strong>vs FD</strong>
   </td>
   <td rowspan="2" ><strong>vs CU baseline</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Baseline</strong>
   </td>
   <td><strong>Opt 6</strong>
   </td>
   <td><strong>Baseline</strong>
   </td>
   <td><strong>Opt 6</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>137
   </td>
   <td>143
   </td>
   <td>111
   </td>
   <td>262
   </td>
   <td>250
   </td>
   <td>322
   </td>
   <td><strong>1.23</strong>
   </td>
   <td><strong>1.29</strong>
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>234
   </td>
   <td>257
   </td>
   <td>192
   </td>
   <td>305
   </td>
   <td>278
   </td>
   <td>372
   </td>
   <td><strong>1.22</strong>
   </td>
   <td><strong>1.34</strong>
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>432
   </td>
   <td>455
   </td>
   <td>346
   </td>
   <td>331
   </td>
   <td>314
   </td>
   <td>414
   </td>
   <td><strong>1.25</strong>
   </td>
   <td><strong>1.32</strong>
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>815
   </td>
   <td>866
   </td>
   <td>642
   </td>
   <td>351
   </td>
   <td>331
   </td>
   <td>446
   </td>
   <td><strong>1.27</strong>
   </td>
   <td><strong>1.35</strong>
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1581
   </td>
   <td>1659
   </td>
   <td>1244
   </td>
   <td>362
   </td>
   <td>345
   </td>
   <td>460
   </td>
   <td><strong>1.27</strong>
   </td>
   <td><strong>1.33</strong>
   </td>
  </tr>
</table>


**Table 9** Performance of Optimization 6 for INT4 GQA (group-wise quantization with num groups = 4)


<table class="table table-bordered">
  <tr>
   <td rowspan="3" ><strong>Batch size</strong>
   </td>
   <td colspan="2" ><strong>Time (us)</strong>
   </td>
   <td colspan="2" ><strong>Bandwidth (GB/s)</strong>
   </td>
   <td><strong>Speed up</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td><strong>CUDA_WMMA</strong>
   </td>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td><strong>CUDA_WMMA</strong>
   </td>
   <td rowspan="2" ><strong>vs FD</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Opt 6</strong>
   </td>
   <td><strong>Opt 6</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>129
   </td>
   <td>116
   </td>
   <td>325
   </td>
   <td>364
   </td>
   <td>1.31
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>219
   </td>
   <td>195
   </td>
   <td>385
   </td>
   <td>431
   </td>
   <td>1.36
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>392
   </td>
   <td>347
   </td>
   <td>429
   </td>
   <td>484
   </td>
   <td>1.39
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>719
   </td>
   <td>638
   </td>
   <td>468
   </td>
   <td>527
   </td>
   <td>1.41
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1375
   </td>
   <td>1225
   </td>
   <td>489
   </td>
   <td>550
   </td>
   <td>1.43
   </td>
  </tr>
</table>



### Optimization 7: Compute max <code class="language-plaintext highlighter-rouge">QK<sup>T</sup></code> From WMMA Fragment Directly (A100/H100 specific)

**Problem Analysis:**

We observe large stalls due to shared memory accessing during the max <code class="language-plaintext highlighter-rouge">QK<sup>T</sup></code> computation (showing as large short scoreboard stalls) as shown in Figure 12.


![Figure 12: Stalls due to shared memory access during max QKT computation](/assets/images/int4-decoding/fg12.png){:style="width:100%"}

**Figure 12** Stalls due to shared memory access during max <code class="language-plaintext highlighter-rouge">QK<sup>T</sup></code> computation (the number that the arrow points to is stall cycles caused by shared memory wait)

**Solution:**

We bypass shared memory when computing max <code class="language-plaintext highlighter-rouge">QK<sup>T</sup></code> by computing it from the WMMA fragment (i.e., the tensor core fragment) directly.  The layout of the WMMA fragment is specific to the GPU architecture.  In this optimization, we only enabled this optimization for the NVIDIA A100/H100 GPUs. Other GPUs will still use shared memory for the max <code class="language-plaintext highlighter-rouge">QK<sup>T</sup></code> computation. By bypassing shared memory, we effectively eliminate the stalls caused by shared memory access.  The tensor core layout of the `C` fragment which is used for storing the <code class="language-plaintext highlighter-rouge">QK<sup>T</sup></code> results is shown in Figure 13.


![Figure 13: C fragment (QKT storage) tensor core layout on A100/H100](/assets/images/int4-decoding/fg13.jpg){:style="width:100%"}

**Figure 13** `C` fragment (<code class="language-plaintext highlighter-rouge">QK<sup>T</sup></code> storage) tensor core layout on A100/H100

**Table 10** Performance of Optimization 7 for INT4 GQA (row-wise quantization)


<table class="table table-bordered">
  <tr>
   <td rowspan="3" ><strong>Batch size</strong>
   </td>
   <td colspan="3" ><strong>Time (us)</strong>
   </td>
   <td colspan="3" ><strong>Bandwidth (GB/s)</strong>
   </td>
   <td colspan="2" ><strong>Speed up</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CU</strong>
   </td>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CU</strong>
   </td>
   <td rowspan="2" ><strong>vs FD</strong>
   </td>
   <td rowspan="2" ><strong>vs CU baseline</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Baseline</strong>
   </td>
   <td><strong>Opt 7</strong>
   </td>
   <td><strong>Baseline</strong>
   </td>
   <td><strong>Opt 7</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>137
   </td>
   <td>143
   </td>
   <td>107
   </td>
   <td>262
   </td>
   <td>250
   </td>
   <td>333
   </td>
   <td><strong>1.27</strong>
   </td>
   <td><strong>1.33</strong>
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>234
   </td>
   <td>257
   </td>
   <td>183
   </td>
   <td>305
   </td>
   <td>278
   </td>
   <td>391
   </td>
   <td><strong>1.28</strong>
   </td>
   <td><strong>1.40</strong>
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>432
   </td>
   <td>455
   </td>
   <td>333
   </td>
   <td>331
   </td>
   <td>314
   </td>
   <td>430
   </td>
   <td><strong>1.30</strong>
   </td>
   <td><strong>1.37</strong>
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>815
   </td>
   <td>866
   </td>
   <td>620
   </td>
   <td>351
   </td>
   <td>331
   </td>
   <td>461
   </td>
   <td><strong>1.31</strong>
   </td>
   <td><strong>1.40</strong>
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1581
   </td>
   <td>1659
   </td>
   <td>1206
   </td>
   <td>362
   </td>
   <td>345
   </td>
   <td>475
   </td>
   <td><strong>1.31</strong>
   </td>
   <td><strong>1.38</strong>
   </td>
  </tr>
</table>


**Table 11** Performance of Optimization 7 for INT4 GQA (group-wise quantization with num groups = 4)


<table class="table table-bordered">
  <tr>
   <td rowspan="3" ><strong>Batch size</strong>
   </td>
   <td colspan="3" ><strong>Time (us)</strong>
   </td>
   <td colspan="3" ><strong>Bandwidth (GB/s)</strong>
   </td>
   <td colspan="2" ><strong>Speed up</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CUDA_WMMA</strong>
   </td>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CUDA_WMMA</strong>
   </td>
   <td rowspan="2" ><strong>vs FD</strong>
   </td>
   <td rowspan="2" ><strong>vs CUDA_WMMA Opt 6</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Opt 6</strong>
   </td>
   <td><strong>Opt 7</strong>
   </td>
   <td><strong>Opt 6</strong>
   </td>
   <td><strong>Opt 7</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>129
   </td>
   <td>116
   </td>
   <td>111
   </td>
   <td>325
   </td>
   <td>364
   </td>
   <td>380
   </td>
   <td><strong>1.17</strong>
   </td>
   <td><strong>1.04</strong>
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>219
   </td>
   <td>195
   </td>
   <td>187
   </td>
   <td>385
   </td>
   <td>431
   </td>
   <td>449
   </td>
   <td><strong>1.17</strong>
   </td>
   <td><strong>1.04</strong>
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>392
   </td>
   <td>347
   </td>
   <td>333
   </td>
   <td>429
   </td>
   <td>484
   </td>
   <td>506
   </td>
   <td><strong>1.18</strong>
   </td>
   <td><strong>1.04</strong>
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>719
   </td>
   <td>638
   </td>
   <td>615
   </td>
   <td>468
   </td>
   <td>527
   </td>
   <td>547
   </td>
   <td><strong>1.17</strong>
   </td>
   <td><strong>1.04</strong>
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1375
   </td>
   <td>1225
   </td>
   <td>1184
   </td>
   <td>489
   </td>
   <td>550
   </td>
   <td>569
   </td>
   <td><strong>1.16</strong>
   </td>
   <td><strong>1.03</strong>
   </td>
  </tr>
</table>



### Optimization 8: Write FP32->BF16 Results to `P` Fragment Directly (A100/H100 specific)

**Problem Analysis:**

During the FP32-BF16 conversion for the `P` fragment, the kernel loads the FP32 data from shared memory, does the conversion and then stores the BF16 data back to shared memory.  Moreover, the conversion requires many thread block synchronizations (`__syncthreads()`).  

**Solution:**

Due to the data partitioning design of the kernel, each warp performs only one pass through the `P` fragment.  Thus, we do not have to write the conversion results back to the shared memory for future usage.  To avoid writing the BF16 data to the shared memory and thread block synchronizations, we have each warp load the FP32 data of the `P` WMMA fragment from the shared memory, do the conversion and then write the BF16 data directly to the `P` fragment. 

Note that this optimization is applied to only the NVIDIA A100 and H100 GPUs because the WMMA fragment layout is architecture dependent.  For non-A100/H100 GPUs, the kernel will fallback to the original path.

The `P` fragment tensor core layout is shown in Figure 14.  Note that this layout is specific to the NVIDIA A100/H100 GPU.

![Figure 14: P fragment tensor core layout on A100/H100](/assets/images/int4-decoding/fg14.jpg){:style="width:100%"}

**Figure 14** `P` fragment tensor core layout on A100/H100

**Table 12** Performance of Optimization 8 for INT4 GQA (row-wise quantization)


<table class="table table-bordered">
  <tr>
   <td rowspan="3" ><strong>Batch size</strong>
   </td>
   <td colspan="3" ><strong>Time (us)</strong>
   </td>
   <td colspan="3" ><strong>Bandwidth (GB/s)</strong>
   </td>
   <td colspan="2" ><strong>Speed up</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CU</strong>
   </td>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CU</strong>
   </td>
   <td rowspan="2" ><strong>vs FD</strong>
   </td>
   <td rowspan="2" ><strong>vs CU baseline</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Baseline</strong>
   </td>
   <td><strong>Opt 8</strong>
   </td>
   <td><strong>Baseline</strong>
   </td>
   <td><strong>Opt 8</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>137
   </td>
   <td>143
   </td>
   <td>101
   </td>
   <td>262
   </td>
   <td>250
   </td>
   <td>353
   </td>
   <td><strong>1.35</strong>
   </td>
   <td><strong>1.41</strong>
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>234
   </td>
   <td>257
   </td>
   <td>174
   </td>
   <td>305
   </td>
   <td>278
   </td>
   <td>410
   </td>
   <td><strong>1.34</strong>
   </td>
   <td><strong>1.47</strong>
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>432
   </td>
   <td>455
   </td>
   <td>317
   </td>
   <td>331
   </td>
   <td>314
   </td>
   <td>451
   </td>
   <td><strong>1.36</strong>
   </td>
   <td><strong>1.43</strong>
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>815
   </td>
   <td>866
   </td>
   <td>590
   </td>
   <td>351
   </td>
   <td>331
   </td>
   <td>485
   </td>
   <td><strong>1.38</strong>
   </td>
   <td><strong>1.47</strong>
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1581
   </td>
   <td>1659
   </td>
   <td>1143
   </td>
   <td>362
   </td>
   <td>345
   </td>
   <td>501
   </td>
   <td><strong>1.38</strong>
   </td>
   <td><strong>1.45</strong>
   </td>
  </tr>
</table>


**Table 13** Performance of Optimization 8 for INT4 GQA (group-wise quantization with num groups = 4)


<table class="table table-bordered">
  <tr>
   <td rowspan="3" ><strong>Batch size</strong>
   </td>
   <td colspan="3" ><strong>Time (us)</strong>
   </td>
   <td colspan="3" ><strong>Bandwidth (GB/s)</strong>
   </td>
   <td colspan="2" ><strong>Speed up</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CUDA_WMMA</strong>
   </td>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CUDA_WMMA</strong>
   </td>
   <td rowspan="2" ><strong>vs FD</strong>
   </td>
   <td rowspan="2" ><strong>vs CUDA_WMMA Opt 6</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Opt 6</strong>
   </td>
   <td><strong>Opt 8</strong>
   </td>
   <td><strong>Opt 6</strong>
   </td>
   <td><strong>Opt 8</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>129
   </td>
   <td>116
   </td>
   <td>106
   </td>
   <td>325
   </td>
   <td>364
   </td>
   <td>396
   </td>
   <td><strong>1.22</strong>
   </td>
   <td><strong>1.09</strong>
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>219
   </td>
   <td>195
   </td>
   <td>180
   </td>
   <td>385
   </td>
   <td>431
   </td>
   <td>467
   </td>
   <td><strong>1.21</strong>
   </td>
   <td><strong>1.08</strong>
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>392
   </td>
   <td>347
   </td>
   <td>319
   </td>
   <td>429
   </td>
   <td>484
   </td>
   <td>528
   </td>
   <td><strong>1.23</strong>
   </td>
   <td><strong>1.09</strong>
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>719
   </td>
   <td>638
   </td>
   <td>596
   </td>
   <td>468
   </td>
   <td>527
   </td>
   <td>565
   </td>
   <td><strong>1.21</strong>
   </td>
   <td><strong>1.07</strong>
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1375
   </td>
   <td>1225
   </td>
   <td>1138
   </td>
   <td>489
   </td>
   <td>550
   </td>
   <td>591
   </td>
   <td><strong>1.21</strong>
   </td>
   <td><strong>1.08</strong>
   </td>
  </tr>
</table>



### Optimization 9: Swizzle P Shared Memory Layouts (A100/H100 specific)

**Problem Analysis:**

We observe large shared memory bank conflicts during `P` loading.  The amount of bank conflict depends on the memory access stride.  For instance, for split-Ks = 32 and max seq length = 8192, we observed that only 4 out of 32 banks are being accessed in parallel (memory access stride = 256).  From Figure 14, when all threads access element 0, threads that have the same `threadIdx.x % 4` access the same bank.


![Figure 15: P fragment in shared memory before swizzling](/assets/images/int4-decoding/fg15.jpg){:style="width:100%"}


**Figure 15** P fragment in shared memory before swizzling

**Solution:**

We shuffle the layout of `P` load/store in the shared memory in such a way that avoids bank conflicts.  In other words, we store the <code class="language-plaintext highlighter-rouge">QK<sup>T</sup></code> results (`C` fragment) and load them (`P` fragment) using the swizzled layout.  Moreover, instead of using the original memory access stride which is dependent on the number of tokens per thread block, we use the fragment's column size as the stride which is constant.  Thus, the load and store of the `P` fragment is always contiguous.

The new layouts for the C and P fragments are shown in Figure 16.  With the new layout, it is guaranteed that 16 banks are being accessed in parallel as shown in Figure 17.


![Figure 16: The swizzled layouts of C and P fragments](/assets/images/int4-decoding/fg16.jpg){:style="width:100%"}

**Figure 16** The swizzled layouts of C and P fragments




![Figure 17: P fragment in shared memory after swizzling](/assets/images/int4-decoding/fg17.jpg){:style="width:100%"}


**Figure 17** P fragment in shared memory after swizzling

**Table 14** Performance of Optimization 9 for INT4 GQA (row-wise quantization)


<table class="table table-bordered">
  <tr>
   <td rowspan="3" ><strong>Batch size</strong>
   </td>
   <td colspan="3" ><strong>Time (us)</strong>
   </td>
   <td colspan="3" ><strong>Bandwidth (GB/s)</strong>
   </td>
   <td colspan="2" ><strong>Speed up</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CU</strong>
   </td>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CU</strong>
   </td>
   <td rowspan="2" ><strong>vs FD</strong>
   </td>
   <td rowspan="2" ><strong>vs CU baseline</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Baseline</strong>
   </td>
   <td><strong>Opt 9</strong>
   </td>
   <td><strong>Baseline</strong>
   </td>
   <td><strong>Opt 9</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>137
   </td>
   <td>143
   </td>
   <td>98
   </td>
   <td>262
   </td>
   <td>250
   </td>
   <td>365
   </td>
   <td><strong>1.39</strong>
   </td>
   <td><strong>1.46</strong>
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>234
   </td>
   <td>257
   </td>
   <td>167
   </td>
   <td>305
   </td>
   <td>278
   </td>
   <td>429
   </td>
   <td><strong>1.41</strong>
   </td>
   <td><strong>1.54</strong>
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>432
   </td>
   <td>455
   </td>
   <td>299
   </td>
   <td>331
   </td>
   <td>314
   </td>
   <td>479
   </td>
   <td><strong>1.45</strong>
   </td>
   <td><strong>1.52</strong>
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>815
   </td>
   <td>866
   </td>
   <td>549
   </td>
   <td>351
   </td>
   <td>331
   </td>
   <td>521
   </td>
   <td><strong>1.48</strong>
   </td>
   <td><strong>1.58</strong>
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1581
   </td>
   <td>1659
   </td>
   <td>1060
   </td>
   <td>362
   </td>
   <td>345
   </td>
   <td>540
   </td>
   <td><strong>1.49</strong>
   </td>
   <td><strong>1.56</strong>
   </td>
  </tr>
</table>


**Table 15** Performance of Optimization 9 for INT4 GQA (group-wise quantization with num groups = 4)


<table class="table table-bordered">
  <tr>
   <td rowspan="3" ><strong>Batch size</strong>
   </td>
   <td colspan="3" ><strong>Time (us)</strong>
   </td>
   <td colspan="3" ><strong>Bandwidth (GB/s)</strong>
   </td>
   <td colspan="2" ><strong>Speed up</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CUDA_WMMA</strong>
   </td>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CUDA_WMMA</strong>
   </td>
   <td rowspan="2" ><strong>vs FD</strong>
   </td>
   <td rowspan="2" ><strong>vs CUDA_WMMA Opt 6</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Opt 6</strong>
   </td>
   <td><strong>Opt 9</strong>
   </td>
   <td><strong>Opt 6</strong>
   </td>
   <td><strong>Opt 9</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>129
   </td>
   <td>116
   </td>
   <td>105
   </td>
   <td>325
   </td>
   <td>364
   </td>
   <td>400
   </td>
   <td><strong>1.23</strong>
   </td>
   <td><strong>1.10</strong>
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>219
   </td>
   <td>195
   </td>
   <td>174
   </td>
   <td>385
   </td>
   <td>431
   </td>
   <td>484
   </td>
   <td><strong>1.26</strong>
   </td>
   <td><strong>1.12</strong>
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>392
   </td>
   <td>347
   </td>
   <td>302
   </td>
   <td>429
   </td>
   <td>484
   </td>
   <td>558
   </td>
   <td><strong>1.30</strong>
   </td>
   <td><strong>1.15</strong>
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>719
   </td>
   <td>638
   </td>
   <td>560
   </td>
   <td>468
   </td>
   <td>527
   </td>
   <td>601
   </td>
   <td><strong>1.28</strong>
   </td>
   <td><strong>1.14</strong>
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1375
   </td>
   <td>1225
   </td>
   <td>1065
   </td>
   <td>489
   </td>
   <td>550
   </td>
   <td>632
   </td>
   <td><strong>1.29</strong>
   </td>
   <td><strong>1.15</strong>
   </td>
  </tr>
</table>



### Optimization 10: Pad Shared Memory for INT4 Dequantization

**Problem Analysis:**

Once the kernel reads the INT4 `K` or `V` cache from global memory, it performs dequantization and stores the results (BF16) in the shared memory.  Then, the BF16 data is loaded to the WMMA fragment from shared memory (via the WMMA interface).  We observed a large number of bank conflicts for both `K` and `V` accesses.  For instance, for `K` stores, only 4 out of 32 banks are being accessed in parallel.  For `K` loads, 16 banks are being accessed in parallel.  The same also occurs for `V` stores and loads.  See the figures in the solution section.

**Solution:**

We pad the shared memory to reduce the bank conflict.  Specifically, we pad each row by 2.  That is, the row stride of `K` becomes `F_K` + 2 and the row stride of V becomes `F_N` + 2 (`F_K` and `F_N` are the fixed widths of the `K` and `V` WMMA fragments, respectively).  With this optimization, we are able to reduce the bank conflict by 1.8x as shown in Figure 18.


![Figure 18: Bank conflicts before and after Optimization 10](/assets/images/int4-decoding/fg18.png){:style="width:100%"}


**Figure 18** Bank conflicts before and after Optimization 10

After Optimization 10, for `K` stores, 32 banks are being accessed in parallel (shown in Figure 19), while for `K` loads, 29 banks are accessed in parallel (shown in Figure 20).

![Figure 19: K fragment store shared memory layout without and with padding](/assets/images/int4-decoding/fg19.jpg){:style="width:100%"}


**Figure 19** K fragment store shared memory layout without and with padding

![Figure 20: K fragment load shared memory layout without and with padding](/assets/images/int4-decoding/fg20.jpg){:style="width:100%"}


**Figure 20** K fragment load shared memory layout without and with padding

**Table 16** Performance of Optimization 10 for INT4 GQA (row-wise quantization)


<table class="table table-bordered">
  <tr>
   <td rowspan="3" ><strong>Batch size</strong>
   </td>
   <td colspan="3" ><strong>Time (us)</strong>
   </td>
   <td colspan="3" ><strong>Bandwidth (GB/s)</strong>
   </td>
   <td colspan="2" ><strong>Speed up</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CU</strong>
   </td>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CU</strong>
   </td>
   <td rowspan="2" ><strong>vs FD</strong>
   </td>
   <td rowspan="2" ><strong>vs CU baseline</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Baseline</strong>
   </td>
   <td><strong>Opt 10</strong>
   </td>
   <td><strong>Baseline</strong>
   </td>
   <td><strong>Opt 10</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>137
   </td>
   <td>143
   </td>
   <td>94
   </td>
   <td>262
   </td>
   <td>250
   </td>
   <td>380
   </td>
   <td><strong>1.45</strong>
   </td>
   <td><strong>1.52</strong>
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>234
   </td>
   <td>257
   </td>
   <td>151
   </td>
   <td>305
   </td>
   <td>278
   </td>
   <td>475
   </td>
   <td><strong>1.55</strong>
   </td>
   <td><strong>1.71</strong>
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>432
   </td>
   <td>455
   </td>
   <td>266
   </td>
   <td>331
   </td>
   <td>314
   </td>
   <td>538
   </td>
   <td><strong>1.63</strong>
   </td>
   <td><strong>1.71</strong>
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>815
   </td>
   <td>866
   </td>
   <td>489
   </td>
   <td>351
   </td>
   <td>331
   </td>
   <td>586
   </td>
   <td><strong>1.67</strong>
   </td>
   <td><strong>1.77</strong>
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1581
   </td>
   <td>1659
   </td>
   <td>930
   </td>
   <td>362
   </td>
   <td>345
   </td>
   <td>616
   </td>
   <td><strong>1.70</strong>
   </td>
   <td><strong>1.79</strong>
   </td>
  </tr>
</table>


**Table 17** Performance of Optimization 10 for INT4 GQA (group-wise quantization with num groups = 4)


<table class="table table-bordered">
  <tr>
   <td rowspan="3" ><strong>Batch size</strong>
   </td>
   <td colspan="3" ><strong>Time (us)</strong>
   </td>
   <td colspan="3" ><strong>Bandwidth (GB/s)</strong>
   </td>
   <td colspan="2" ><strong>Speed up</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CUDA_WMMA</strong>
   </td>
   <td rowspan="2" ><strong>FD</strong>
   </td>
   <td colspan="2" ><strong>CUDA_WMMA</strong>
   </td>
   <td rowspan="2" ><strong>vs FD</strong>
   </td>
   <td rowspan="2" ><strong>vs CUDA_WMMA Opt 6</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Opt 6</strong>
   </td>
   <td><strong>Opt 10</strong>
   </td>
   <td><strong>Opt 6</strong>
   </td>
   <td><strong>Opt 10</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>129
   </td>
   <td>116
   </td>
   <td>99
   </td>
   <td>325
   </td>
   <td>364
   </td>
   <td>425
   </td>
   <td><strong>1.31</strong>
   </td>
   <td><strong>1.17</strong>
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>219
   </td>
   <td>195
   </td>
   <td>161
   </td>
   <td>385
   </td>
   <td>431
   </td>
   <td>523
   </td>
   <td><strong>1.36</strong>
   </td>
   <td><strong>1.21</strong>
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>392
   </td>
   <td>347
   </td>
   <td>282
   </td>
   <td>429
   </td>
   <td>484
   </td>
   <td>598
   </td>
   <td><strong>1.39</strong>
   </td>
   <td><strong>1.23</strong>
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>719
   </td>
   <td>638
   </td>
   <td>509
   </td>
   <td>468
   </td>
   <td>527
   </td>
   <td>662
   </td>
   <td><strong>1.41</strong>
   </td>
   <td><strong>1.25</strong>
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1375
   </td>
   <td>1225
   </td>
   <td>965
   </td>
   <td>489
   </td>
   <td>550
   </td>
   <td>698
   </td>
   <td><strong>1.43</strong>
   </td>
   <td><strong>1.27</strong>
   </td>
  </tr>
</table>



## Performance Evaluation


### Microbenchmark results

We also evaluated BF16 GQA performance using our optimized kernel (as shown in Table 19).  CU still performs generally worse than FD and FA for BF16.  This is expected since our optimizations are INT4 focused.

While INT4 GQA is still not as efficient as BF16 GQA (see the achieved bandwidths), it is important to note that when comparing FD BF16 GQA performance against CU INT4 GQA performance, **we can see that the latency of INT4 is smaller than that of BF16**.

**Table 19** Performance of BF16 GQA and INT GQA after CU optimizations  

**On A100**


<table class="table table-bordered">
  <tr>
   <td><strong>Time (us)</strong>
   </td>
   <td colspan="4" ><strong>BF16 GQA</strong>
   </td>
   <td colspan="4" ><strong>INT4 GQA</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Batch size</strong>
   </td>
   <td><strong>FD</strong>
   </td>
   <td><strong>FA</strong>
   </td>
   <td><strong>CU before</strong>
   </td>
   <td><strong>CU after</strong>
   </td>
   <td><strong>FD</strong>
   </td>
   <td><strong>FA</strong>
   </td>
   <td><strong>CU before</strong>
   </td>
   <td><strong>CU after</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>139
   </td>
   <td>133
   </td>
   <td>183
   </td>
   <td>163
   </td>
   <td>137
   </td>
   <td>-
   </td>
   <td>143
   </td>
   <td>94
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>245
   </td>
   <td>229
   </td>
   <td>335
   </td>
   <td>276
   </td>
   <td>234
   </td>
   <td>-
   </td>
   <td>257
   </td>
   <td>151
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>433
   </td>
   <td>555
   </td>
   <td>596
   </td>
   <td>517
   </td>
   <td>432
   </td>
   <td>-
   </td>
   <td>455
   </td>
   <td>266
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>826
   </td>
   <td>977
   </td>
   <td>1127
   </td>
   <td>999
   </td>
   <td>815
   </td>
   <td>-
   </td>
   <td>866
   </td>
   <td>489
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1607
   </td>
   <td>1670
   </td>
   <td>2194
   </td>
   <td>1879
   </td>
   <td>1581
   </td>
   <td>-
   </td>
   <td>1659
   </td>
   <td>930
   </td>
  </tr>
</table>



<table class="table table-bordered">
  <tr>
   <td><strong>Effective Bandwidth (GB/s)</strong>
   </td>
   <td colspan="4" ><strong>BF16 GQA</strong>
   </td>
   <td colspan="4" ><strong>INT4 GQA</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Batch size</strong>
   </td>
   <td><strong>FD</strong>
   </td>
   <td><strong>FA</strong>
   </td>
   <td><strong>CU before</strong>
   </td>
   <td><strong>CU after</strong>
   </td>
   <td><strong>FD</strong>
   </td>
   <td><strong>FA</strong>
   </td>
   <td><strong>CU before</strong>
   </td>
   <td><strong>CU after</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>965
   </td>
   <td>1012
   </td>
   <td>736
   </td>
   <td>824
   </td>
   <td>262
   </td>
   <td>-
   </td>
   <td>250
   </td>
   <td>380
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>1097
   </td>
   <td>1175
   </td>
   <td>802
   </td>
   <td>972
   </td>
   <td>305
   </td>
   <td>-
   </td>
   <td>278
   </td>
   <td>475
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>1240
   </td>
   <td>968
   </td>
   <td>901
   </td>
   <td>1039
   </td>
   <td>331
   </td>
   <td>-
   </td>
   <td>314
   </td>
   <td>538
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>1301
   </td>
   <td>1100
   </td>
   <td>954
   </td>
   <td>1075
   </td>
   <td>351
   </td>
   <td>-
   </td>
   <td>331
   </td>
   <td>586
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1338
   </td>
   <td>1287
   </td>
   <td>980
   </td>
   <td>1144
   </td>
   <td>362
   </td>
   <td>-
   </td>
   <td>345
   </td>
   <td>616
   </td>
  </tr>
</table>


**On H100**


<table class="table table-bordered">
  <tr>
   <td><strong>Time (us)</strong>
   </td>
   <td colspan="4" ><strong>BF16 GQA</strong>
   </td>
   <td colspan="4" ><strong>INT4 GQA</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Batch size</strong>
   </td>
   <td><strong>FD</strong>
   </td>
   <td><strong>FA</strong>
   </td>
   <td><strong>CU before</strong>
   </td>
   <td><strong>CU after</strong>
   </td>
   <td><strong>FD</strong>
   </td>
   <td><strong>FA</strong>
   </td>
   <td><strong>CU before</strong>
   </td>
   <td><strong>CU after</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>91
   </td>
   <td>90
   </td>
   <td>114
   </td>
   <td>100
   </td>
   <td>70
   </td>
   <td>-
   </td>
   <td>96
   </td>
   <td>64
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>148
   </td>
   <td>146
   </td>
   <td>200
   </td>
   <td>183
   </td>
   <td>113
   </td>
   <td>-
   </td>
   <td>162
   </td>
   <td>101
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>271
   </td>
   <td>298
   </td>
   <td>361
   </td>
   <td>308
   </td>
   <td>205
   </td>
   <td>-
   </td>
   <td>294
   </td>
   <td>170
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>515
   </td>
   <td>499
   </td>
   <td>658
   </td>
   <td>556
   </td>
   <td>389
   </td>
   <td>-
   </td>
   <td>558
   </td>
   <td>306
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>1000
   </td>
   <td>1011
   </td>
   <td>1260
   </td>
   <td>1066
   </td>
   <td>756
   </td>
   <td>-
   </td>
   <td>1066
   </td>
   <td>575
   </td>
  </tr>
</table>



<table class="table table-bordered">
  <tr>
   <td><strong>Effective Bandwidth (GB/s)</strong>
   </td>
   <td colspan="4" ><strong>BF16 GQA</strong>
   </td>
   <td colspan="4" ><strong>INT4 GQA</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Batch size</strong>
   </td>
   <td><strong>FD</strong>
   </td>
   <td><strong>FA</strong>
   </td>
   <td><strong>CU before</strong>
   </td>
   <td><strong>CU after</strong>
   </td>
   <td><strong>FD</strong>
   </td>
   <td><strong>FA</strong>
   </td>
   <td><strong>CU before</strong>
   </td>
   <td><strong>CU after</strong>
   </td>
  </tr>
  <tr>
   <td>32
   </td>
   <td>1481
   </td>
   <td>1496
   </td>
   <td>1178
   </td>
   <td>1341
   </td>
   <td>511
   </td>
   <td>-
   </td>
   <td>371
   </td>
   <td>560
   </td>
  </tr>
  <tr>
   <td>64
   </td>
   <td>1815
   </td>
   <td>1840
   </td>
   <td>1345
   </td>
   <td>1470
   </td>
   <td>631
   </td>
   <td>-
   </td>
   <td>443
   </td>
   <td>710
   </td>
  </tr>
  <tr>
   <td>128
   </td>
   <td>1982
   </td>
   <td>1802
   </td>
   <td>1487
   </td>
   <td>1743
   </td>
   <td>699
   </td>
   <td>-
   </td>
   <td>487
   </td>
   <td>844
   </td>
  </tr>
  <tr>
   <td>256
   </td>
   <td>2087
   </td>
   <td>2156
   </td>
   <td>1634
   </td>
   <td>1934
   </td>
   <td>736
   </td>
   <td>-
   </td>
   <td>513
   </td>
   <td>935
   </td>
  </tr>
  <tr>
   <td>512
   </td>
   <td>2150
   </td>
   <td>2127
   </td>
   <td>1706
   </td>
   <td>2015
   </td>
   <td>757
   </td>
   <td>-
   </td>
   <td>537
   </td>
   <td>996
   </td>
  </tr>
</table>



### E2E results

We evaluated our optimized INT4 GQA kernel in Llama 2 70B on 8 H100 GPUs. We ran the model end-to-end, but only reported the decode latency.  We use FP8 FFN (feed forward network) to emphasize the attention performance in the decoding phase.  We vary the batch size from 1 to 256 and the context length from 2,048 (2K) to 16,384 (16K).  The E2E performance results are shown in the figure below.

![Figure 21: Meta Llama 2 decode latency (ms) comparison](/assets/images/int4-decoding/fg21.png){:style="width:100%"}


**Figure 21** Meta Llama 2 decode latency (ms) comparison (BF16 GQA runs out of memory in large batch size configurations)


## Code

If you are interested, please checkout our code [here](https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/gen_ai).  If you have any questions, please feel free to open an issue on GitHub, and we will be happy to help.  Your contributions are welcome!