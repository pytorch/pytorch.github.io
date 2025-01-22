---
layout: blog_detail
title: "Accelerating LLM Inference with GemLite, TorchAO and SGLang"
author: "Teams at PyTorch, Mobius Labs and SGLang"  
---

Large Language Models (LLMs) are typically very resource-intensive, requiring significant amounts of memory, compute and power to operate effectively. Quantization provides a solution by reducing weights and activations from 16 bit floats to lower bitrates (e.g., 8 bit, 4 bit, 2 bit), achieving significant speedup and memory savings and also enables support for larger batch sizes.

Existing solutions for low precision inference work well for small batch sizes, but suffer from following issues:

* Performance drops when we increase the batch size
* Restrictions on types of quantization, for example, some kernels only support symmetric quantization that could have implications on accuracy of the model at lower bits
* Interplay between quantization, serialization, and tensor parallelism (TP) makes it difficult to load quantized models and requires changes to user models

To address these challenges, we created an end-to-end, performant, modular and extensible low-precision inference solution integrating the following libraries:

* [GemLite](https://github.com/mobiusml/gemlite), a Triton kernel library, tackles the performance limitations of large batch sizes and restrictions on the types of quantization
* [TorchAO](https://github.com/pytorch/ao), a PyTorch-native library, provides a streamlined experience for quantization, sparsity, and tensor parallelism (with DTensor)
* [SGLang](https://github.com/sgl-project/sglang), a fast, efficient and hackable serving framework for Large Language Model (LLM) and Vision Language Models (VLM) with extensive model support

If you’re interested in trying this out in SGLang, please follow these [repro instructions](#repro-instructions). For the rest of the blog, we’ll walk through relevant details for GemLite, TorchAO and SGlang both in terms of the design of the library itself and integration in addressing the problems we mentioned above, in the end we’ll present the benchmarking results on Llama 3.1-8B model across different batch sizes and tensor parallel sizes.

## 1. Teaser of Results 

Following is a summary of the results in 8xH100 machine on Llama 3.1-8B for decode. For all experiments, the baseline is bfloat16 torch.compiled model:


<table class="table table-bordered">
  <tr>
   <td>
   </td>
   <td>bfloat16 w/ torch.compile
   </td>
   <td>int4 weight only quantization, group size 64
   </td>
   <td>float8 per row dynamic quantization
   </td>
  </tr>
  <tr>
   <td>Batch size 1, TP size 1
   </td>
   <td>131 tokens/sec
   </td>
   <td>255 tokens/sec (1.95x speedup)
   </td>
   <td>166 tokens/sec (1.27x speedup)
   </td>
  </tr>
  <tr>
   <td>Batch size 32, TP size 1
   </td>
   <td>2799 tokens/sec
   </td>
   <td>3241 tokens/sec (1.16x speedup)
   </td>
   <td>3586 tokens/sec (1.28x speedup)
   </td>
  </tr>
  <tr>
   <td>Batch size 32, TP size 4
   </td>
   <td>5575 tokens/sec
   </td>
   <td>6334 tokens/sec (1.14x speedup)
   </td>
   <td>6159 tokens/sec (1.10x speedup)
   </td>
  </tr>
</table>


Our solution supports NVIDIA GPUs, including H100 and A100, and achieves speedup over the compiled bfloat16 baseline across batch sizes and TP sizes for both int4 weight only (from 1.14x to 1.95x) and float8 dynamic quantization (from 1.10x to 1.28x). Note that quantization may have a small impact on accuracy, which is outside the scope of this blogpost. Our int4 weight-only quantization is compatible with accuracy preserving techniques like HQQ. Please refer to [TorchAO's README](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md#cuda-backend-1), [this benchmark](https://huggingface.co/mobiuslabsgmbh/Llama-3.1-8b-instruct_4bitgs64_hqq_calib), and [this blog](https://neuralmagic.com/blog/we-ran-over-half-a-million-evaluations-on-quantized-llms-heres-what-we-found/) for more information.


## 2. GemLite: Kernel Development

The kernels were developed as part of GemLite, a project dedicated to optimizing low-bit matrix multiplication kernels. Developed using Triton, GemLite provides highly flexible and performant solutions across various activations, bitrates and hardware. In a nutshell, the kernels offer:



* Support for various activation data types:  fp16, int8 and fp8
* Compatibility: works seamlessly with non-packed (e.g., int8, fp8) and packed formats (e.g., uint4, uint2, uint1)
* Performance Optimization: includes optimized kernels and autotuning tools to achieve high performance across different hardware and batch sizes
* Integration: Compatible with torch.compile and CUDA graphs, ensuring support for advanced features like tensor parallelism

### Kernel Selection

Optimizing kernel selection for large language model (LLM) generation requires addressing the distinct needs of different batch sizes. LLM workloads involve a mix of compute-bound and memory-bound iterations: smaller batch sizes are memory-bound, while larger batch sizes become compute-bound. GemLite kernels are designed to adapt to these varying demands, ensuring optimal execution for each scenario.

In memory-bound scenarios, where data transfer is the limiting factor, the processor often waits for data to be fetched, leading to underutilized computational resources. For batch size = 1, a GEMV kernel performs best, whereas for larger batch sizes, GEMM kernels are more efficient. For batch sizes between 2 and 64, when matrices are "skinny," a GEMM-SPLITK kernel is used to enable better GPU utilization ([arXiv](https://arxiv.org/abs/2402.00025)).

GemLite includes the following kernels optimized for each of these scenarios:

### Single Sample Inference

For single-sample inferences, we use GEMV kernels. However, asymmetric quantization methods require additional metadata, such as scales and zero points, to be loaded for each block. This can lead to increased memory transfer, so careful handling is essential.

Specifically, for packed data, our experiments indicate that loading scales and zero points only once per two consecutive blocks minimizes redundant operations. Since these blocks share the same metadata, this approach results in:

* 5–8% end-to-end inference speedup compared to the default GEMV kernel
* 30–40% improvement over the traditional Split-K method

This new kernel/algorithm, GEMV_REVSPLITK, is available [here](https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/gemv_revsplitK_A16fWnO16f_int32packing.py).

For non-packed data, the [GEMV_SPLITK](https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/gemv_splitK_A16fWnO16f_int32packing.py) algorithm is employed. This algorithm iterates over the k-dimension to compute the dot product without relying on Triton's tl.dot.

### Batched Inference

For moderate batch sizes, we use the GEMM-based Split-K method ([arXiv](https://arxiv.org/abs/2402.00025)) which splits the k-dimension (weight rows) into multiple jobs. The optimal-split SPLIT_K parameter is found by autotuning values ranging from 1 to 16. Setting SPLIT_K=1 enables a fallback implementation to a GEMM kernel, allowing the same kernel code to be used for compute-bound batch sizes starting from 32 and 64, depending on the matrix shape and the device.

### Maximizing High Performance: Key Implementation Insights

Various implementation details must be carefully addressed to achieve high performance. Following are some of the key aspects we focused on to ensure high performance:

1. Autotuning for Performance


    [Autotuning](https://triton-lang.org/main/python-api/generated/triton.autotune.html) is critical for achieving optimal kernel performance. Since this process can be time-intensive, GemLite provides tools to automatically save and load autotuning results for all kernels. This ensures that the autotuning process is performed only once per GPU device, minimizing runtime, reducing repetitive overhead, and maintaining consistent performance across runs.

2.  Ensuring Kernel Correctness


    Ensuring kernel correctness across different quantization and configuration settings is essential. Triton’s [early configuration pruning](https://triton-lang.org/main/python-api/generated/triton.autotune.html) plays a key role in this process. For example, during Split-K tuning, configurations are selected only if K is divisible by BLOCK_SIZE_K × SPLIT_K,, and BLOCKS_SIZE_K  is further pruned based on the group-size value. This approach ensures both efficiency and correctness in kernel operation.

3. Overcoming Bit-Unpacking Bottlenecks


    When deploying on data center-grade GPUs like NVIDIA’s A100 and H100, performance bottlenecks related to bit-unpacking were observed. To mitigate these, various bit-packing configurations were explored, including packing along columns versus rows and experimenting with different bit-packing widths (e.g., 8-bit vs. 32-bit). Notably, transitioning from 32-bit to 8-bit packing delivered performance improvements of up to 18% on the A100 and 6% on the H100

4. torch.compile compatibility


    To ensure seamless compatibility with PyTorch’s torch.compile, kernel calls are wrapped in a [custom_op](https://pytorch.org/tutorials/advanced/python_custom_ops.html). This integration allows advanced features such as pre-hooks and early configuration pruning to function correctly, delivering accurate results without sacrificing performance. While some of these [features](https://github.com/pytorch/pytorch/issues/139059) are not yet fully supported in PyTorch, the custom_op implementation effectively bridges the gap, ensuring smooth integration and high performance. 


## 3. TorchAO

TorchAO is a PyTorch native quantization and sparsity library for both training and inference, featuring simple user APIs to train, quantize and deploy low precision models, and composability with other PyTorch features like distributed inference and torch.compile.

PyTorch does not support low precision dtypes or different packing formats by default. With Tensor Subclass, we extend PyTorch native Tensor abstractions and model quantization as dtype conversion, while different packing formats for custom kernels are handled through layouts. For example, we support quantized linear operations with int4 weights, packed in a Tensor Core friendly layout, with tinygemm or GemLite kernel implementations. More details can be found [here](https://pytorch.org/ao/stable/contributor_guide.html).


![flow diagram](/assets/images/accelerating-llm-inference/fg1.png){:style="width:100%"}


Apart from more PyTorch native abstractions for developers, we want to highlight two benefits of this design for modeling users.

1. [Serialization](https://pytorch.org/ao/stable/serialization.html): Save and load quantized weights into a state_dict just like a floating point model, eliminating the need to transform floating point model to quantized model before the quantized weights are loaded. This reduces friction of distributing and deploying quantized models.

2. [Composability](#torch-tensor-parallel): Seamless integration with downstream features like tensor parallel, allowing users to focus on modeling without worrying about compatibility with tensor parallel, torch.compile, and other PyTorch features. Since these features are implemented with Tensor level abstraction, users can quantize and do distributed inference with no model changes most of the time.


### GemLite Kernel Integration

To achieve the aforementioned benefits for the GemLite kernel, we integrated GemLite into TorchAO. This integration takes advantage of GemLite’s wide support and flexibility to allow for weight only quantization at 4 and 8 bits, under asymmetric and symmetric quantization schemes, 32 and 8 bit packing sizes, as well as grouped and ungrouped quantization. We enable this integration via the  `quantize_` api which can be used alongside the GemLite constructor as follows


```
quantize_(model, gemlite_uintx_weight_only(group_size, bit_width, packing_bitwidth))
```


The primary difficulty in creating this integration was making sure that the TorchAO composability guarantees were satisfied for the entire breadth of GemLite quantization kernel options. While the primary integration was relatively straight forward, making sure every different quantization type and their associated kernels worked well with tensor parallel was non-trivial.


### Torch Tensor Parallel {#torch-tensor-parallel}

Tensor Parallelism is an effective way to speed up LLM inference. TP shards large matrices of linear or embedding modules onto multiple devices, typically in column-wise or row-wise styles. As the weight matrix gets distributed, computation is decomposed too. For example, the column-wise pattern below enables simultaneous matrix-vector multiply on four devices:

![equation](/assets/images/accelerating-llm-inference/fg5.jpg){:style="max-width:300px; width:100%; display: block; margin-left: auto; margin-right: auto"}
 

PyTorch implements TP by converting a regular tensor (e.g. matrix *A*) into a *DTensor*:

```
dtensor = _shard_tensor(mA, device_mesh, (Shard(0),))
```

Since DTensor stores meta information about the sharding, it knows how to reconstruct the full result when needed. Take Transformers’ feedforward module for example, as the down projection and up projection use column-wise and row-wise sharding respectively, DTensor will automatically perform an all-reduce on the ranks’ results as they move into the next operation. Such automation allows model authors to focus on computation without worrying about the communication needed for distributed execution.

**Tensor Parallel and Quantization Order**

Since both DTensor and quantization are tensor-level transformations, the application order matters in ensuring a workflow can generally work on different setups. We have two observations: (i) checkpoints are typically saved in quantized formats, to save the quantization overhead before each run; and (ii) TP may run on a different number of devices, depending on resource constraints or service agreements. As such, we first apply quantization to the original tensor, save it to disk depending on whether a reuse is desired. At service launch time, we load the quantized checkpoint and shard the tensors into DTensors on-the-fly as we load them into the model.

**Tensor Parallel Support in TorchAO**

Since we quantize the model first then distribute the Tensor, we’ll have `DTensor(QuantizedTensor(weight))`, where `DTensor` means a distributed Tensor class and `QuantizedTensor` means a quantized tensor class in TorchAO. `QuantizedTensor` should support the operators called when constructing a `DTensor`, including slice and view ops. To make sure the overall execution is efficient, the packed weight that’s sliced in the dimension 0 and 1 should match the result of first slice the unpacked weight then pack (pack and slice operation should commute), otherwise the packing format is not compatible with tensor parallelism.


## 4. SGLang

SGLang is a fast serving framework for large language models and vision language models. It is known for its almost [zero-overhead batch scheduler](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) and fast [constrained decoding](https://lmsys.org/blog/2024-02-05-compressed-fsm/). It is mainly implemented in Python, lightweight, and easy to hack. It is also one of the first frameworks to integrate torch.compile.

**TorchAO integration in SGLang**

We integrated `quantize_` API for applying a specific type of quantization to model into SGLang that supports int4 weight only quantization (both tinygemm and GemLite version), float8 dynamic quantization and a few other types of quantization so far. Users can enable quantization by adding `--torchao-config` argument to the benchmarking script. The currently enabled options also support tensor parallelism through composition with DTensor that is enabled with `--tp-size` option.

**Torch Native Tensor Parallel Support in SGLang**

Existing model definitions in SGLang use special linear modules that are coupled with tensor parallelism style, for example: `MergedColumnParallelLinear`, `QKVParallelLinear` and `RowParallelLinear`. To decouple the model definition and tensor parallelization style, we defined a [pytorch native model](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/torch_native_llama.py) that uses plain `nn.Linear` module from PyTorch and rely on PyTorch tensor parallelism APIs for parallelization and torch.compile for speedup. At related module hierarchies, we add a dictionary describing how a submodule should be parallelized. For example, in `class LlamaAttention`, we define:

```
_tp_plan = {
    "qkv_proj": "Colwise_Sharded",
    "o_proj": "Rowwise",
}
```

where `"qkv_proj" `and `"o_proj" `are the FQNs of the `wqkv` and `wo` projections, and the values are their TP styles. 

We then define a TP engine in `model_parallel.py`. It searches for `_tp_plan `recursively within the model, and applies the indicated TP styles to the submodules using PyTorch’s [parallelize_module](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.parallelize_module) API.


## 5. Results

The evaluation focused on two popular quantization techniques for H100 machines: int4 weight-only quantization and float8 dynamic quantization. These methods were chosen due to their widespread use in optimizing memory efficiency and computational performance on H100 machines, making them ideal candidates for benchmarking against various workloads.



* **int4 Weight-Only Quantization**: This method significantly reduces memory footprint and accelerates decode for memory-bound workloads, with minimal impact on performance in compute-intensive scenarios like prefill or larger batch sizes. We present results for bf16, GemLite, and tinygemm kernels below, across various batch sizes and tensor parallel configurations 
* **float8 Dynamic Quantization**: While offering less memory savings, this method often provides higher accuracy and balanced speedups for both memory-bound and compute-bound tasks. With Hopper-grade hardware and native fp8 support, the efficient cutlass/cuBLAS kernels used by AO contribute to a significant speedup

The graphs below show the decode tokens/sec for different tp sizes, each graph shows the results across different batch sizes and for different types of quantization:



* BF16 is our bfloat16, torch.compile’d baseline
* tinygemm-4-64 is using `int4_weight_only` quantization in TorchAO, it’s a 4 bit groupwise quantization with group size of 64, using tinygemm kernel
* gemlite-4-64 is using `gemlite_uintx_weight_only `quantization in TorchAO, 4 means 4 bit, and 64 is also the group size, using GemLite kernel
* fp8dq-per_row is using `float8_dynamic_activation_float8_weight` quantization in TorchAO, both activation and weights are quantized with per row scales

![bar chart](/assets/images/accelerating-llm-inference/fg2.png){:style="width:100%"}

![bar chart](/assets/images/accelerating-llm-inference/fg3.png){:style="width:100%"}

![bar chart](/assets/images/accelerating-llm-inference/fg4.png){:style="width:100%"}


For int4 weight-only quantization, at batch size 1, the tinygemm kernel achieved the best performance. However, its efficiency declined with increasing batch sizes. Conversely, GemLite effectively bridged this gap, delivering superior performance at larger batch sizes. GemLite also achieved a 9–10x speedup during the prefill phase compared to tinygemm, despite ongoing performance optimizations constrained by Triton.

Float8 dynamic quantization showed 1.3x speedup over bfloat16 consistently with tensor parallel size 1 across different batch sizes and 1.1x to 1.2x speedup in larger tensor parallel sizes. As the tensor parallel size increases, the overall speedup decreases, which is expected due to the reduction in matmul size. Note that we do expect to get speedup for prefill as well, but since we rely on torch.compile for speedup and prefill compile is not enabled in SGLang yet, we will leave this for future work.


### Repro Instructions {#repro-instructions}

We conducted benchmarks on an 8xH100 machine using GemLite 0.4.1, SGLang built from commit feb2b76, TorchAO nightly 0.8.0.dev20241223+cu124, and PyTorch 2.5.1. The Llama-3.1 Instruct models were chosen as the architecture for evaluation. 

```
BATCH_SIZE=16
# Note: gemlite is only compatible with float16
# while int4wo-64 (tinygemm-4-64 as shown in the graph) and fp8dq-per_row should use bfloat16
DTYPE=float16
# int4wo-64, fp8dq-per_tensor
TORCHAO_CONFIG=gemlite-4-64
TP_SIZE=2
# Decode performance
python3 -m sglang.bench_offline_throughput --model-path meta-llama/Llama-3.1-8B-Instruct --json-model-override-args '{"architectures": ["TorchNativeLlamaForCausalLM"]}' --dataset-name random --random-input 1024 --random-output 512 --random-range 1 --num-prompts $BATCH_SIZE --enable-torch-compile --dtype $DTYPE --torchao-config $TORCHAO_CONFIG --tp-size $TP_SIZE

# Example output
# Benchmark...
# [2024-12-20 12:42:16 TP0] Prefill batch. #new-seq: 2, #new-token: 2046, #cached-token: 4, cache hit rate: \0.06%, token usage: 0.00, #running-req: 0, #queue-req: 0
# ...
# [2024-12-20 12:45:35 TP0] Decode batch. #running-req: 16, #token: 16763, token usage: 0.01, gen throughput\ (token/s): 2.20, #queue-req: 0
# [2024-12-20 12:45:38 TP0] Decode batch. #running-req: 16, #token: 24443, token usage: 0.02, gen throughput\ (token/s): 2739.89, #queue-req: 0

# We reported the last throughput (token/s) as the performance for decode
```

## Conclusion

With performant and extensible kernels from [GemLite](https://github.com/mobiusml/gemlite), PyTorch native architecture optimization library [TorchAO](https://github.com/pytorch/ao) and high performance inference framework [SGLang](https://github.com/sgl-project/sglang), we showcased fast end-to-end quantized inference for both int4 and float8 across different batch sizes and tensor parallel sizes with simple and composable user APIs to reduce the resource requirement for LLMs. This integration is our first step towards meeting the needs of fast inference across different models, workloads, precisions and hardwares and we are looking forward to continuing advancing the state of the art for end to end mixed and low precision LLM inference.

Our immediate future work focuses on the following:



* Exploring diverse combinations of weight and activation quantization to strike the best balance between speed and accuracy
* Extending support to additional GPU architectures to broaden accessibility
* Enhancing compatibility with MoE models to address growing demands in scalable inference
* Allow for easy integration of fast custom kernels in TorchAO so that they can be easily leveraged by SGLang and other inference frameworks
* While we didn’t measure accuracy impact in this blogpost, we can develop auto quantization tool in TorchAO to allow users to trade off between performance and accuracy
* Better integration with tensor parallelism in SGLang to support running larger models
* Enable torch.compile for prefill phase in SGLang

We also invite the community to actively test, provide feedback, and contribute to shaping the future of fast and efficient LLM inference.