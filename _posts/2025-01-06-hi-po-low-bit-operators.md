---
layout: blog_detail
title: "High-Performance Low-Bit Operators for PyTorch"
author: Scott Roy, Digant Desai, Kimish Patel
---

We are excited to announce the addition of embedding operators with low-bit weights (1-8 bit) and linear operators with 8-bit dynamically quantized activations and low-bit weights (1-8 bit) for Arm CPUs in TorchAO, PyTorch’s native low-precision library. These operators work seamlessly across all PyTorch surfaces, including eager, torch.compile, AOTI, and ExecuTorch, and are [available to use in torchchat](https://github.com/pytorch/torchchat/blob/main/docs/quantization.md#experimental-torchao-lowbit-kernels).

In developing these linear operators, our focus was on **code sharing between PyTorch and ExecuTorch**, and establishing a clear boundary between the higher-level operator and the lower-level kernel. This design **allows third-party vendors to easily swap in their own kernels**. We also set out to **create a place and infrastructure to experiment** with new CPU quantization ideas and test those across the PyTorch ecosystem.


## Universal low-bit kernels

There is no hardware support for low-bit arithmetic. In what we call universal kernels, we explicitly separated the logic that unpacks low-bit values to int8 values, and the int8 GEMV kernel logic in a modular fashion. We started with an 8-bit kernel, for example, this [1x8 8-bit GEMV kernel](https://github.com/pytorch/ao/blob/299aacd0ab0e0cce376f56e18e5bb585d517b2e1/torchao/experimental/kernels/cpu/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot-impl.h#L64) that uses the Arm neondot instruction. Within the 8-bit kernel, we invoke an [inlined unpacking routine](https://github.com/pytorch/ao/blob/299aacd0ab0e0cce376f56e18e5bb585d517b2e1/torchao/experimental/kernels/cpu/aarch64/linear/channelwise_8bit_activation_groupwise_lowbit_weight_1x8x16_f32_neondot-impl.h#L169) to convert low-bit values into int8 values. This unpacking routine is force-inlined and templated on some low-bit value. Our experiments showed no performance difference between using a separate force-inlined unpacking routine and directly embedding the unpacking code inline.

The advantage of this modular design is improved development speed and code maintainability. After writing an 8-bit kernel, we quickly achieved full low-bit coverage by writing [simple bitpacking routines](https://github.com/pytorch/ao/tree/299aacd0ab0e0cce376f56e18e5bb585d517b2e1/torchao/experimental/kernels/cpu/aarch64/bitpacking). In fact, developers who worked on the bit packing routines did not need to be experts on GEMV/GEMM kernel writing. We also reused the same bitpacking routines from the linear kernels [within the embedding kernels](https://github.com/pytorch/ao/blob/299aacd0ab0e0cce376f56e18e5bb585d517b2e1/torchao/experimental/kernels/cpu/aarch64/embedding/embedding.h#L161). In future we could reuse the same bitpacking routines for universal GEMM kernels or kernels based on fma or i8mm instructions.


## Shared code between PyTorch and ExecuTorch

To achieve shared code between PyTorch and ExecuTorch, we wrote kernels [using raw pointers instead of PyTorch tensors](https://github.com/pytorch/ao/blob/299aacd0ab0e0cce376f56e18e5bb585d517b2e1/torchao/experimental/kernels/cpu/aarch64/linear/linear.h). Moreover, we implemented the [linear operator in a header ](https://github.com/pytorch/ao/blob/299aacd0ab0e0cce376f56e18e5bb585d517b2e1/torchao/experimental/ops/linear_8bit_act_xbit_weight/op_linear_8bit_act_xbit_weight-impl.h#L259)that is included in separate [PyTorch](https://github.com/pytorch/ao/blob/299aacd0ab0e0cce376f56e18e5bb585d517b2e1/torchao/experimental/ops/linear_8bit_act_xbit_weight/op_linear_8bit_act_xbit_weight_aten.cpp) and [ExecuTorch](https://github.com/pytorch/ao/blob/299aacd0ab0e0cce376f56e18e5bb585d517b2e1/torchao/experimental/ops/linear_8bit_act_xbit_weight/op_linear_8bit_act_xbit_weight_executorch/w4s.cpp) operator registration code. By using only features common to both ATen and ExecuTorch tensors, we ensured compatibility between the two frameworks. For multi-threaded compute, we introduced [torchao::parallel_1d](https://github.com/pytorch/ao/blob/299aacd0ab0e0cce376f56e18e5bb585d517b2e1/torchao/experimental/ops/parallel.h#L13), which compiles to either [at::parallel_for](https://github.com/pytorch/ao/blob/299aacd0ab0e0cce376f56e18e5bb585d517b2e1/torchao/experimental/ops/parallel-aten-impl.h) or [ExecuTorch’s threadpool](https://github.com/pytorch/ao/blob/299aacd0ab0e0cce376f56e18e5bb585d517b2e1/torchao/experimental/ops/parallel-executorch-impl.h) based on compile-time flags.


## Swappable kernels

Our design for the higher-level multi-threaded linear operator is agnostic to the lower-level single-threaded kernels, allowing third-party vendors to swap in their own implementations. The interface between the operator and kernel is defined by a [ukernel config](https://github.com/pytorch/ao/blob/299aacd0ab0e0cce376f56e18e5bb585d517b2e1/torchao/experimental/ops/linear_8bit_act_xbit_weight/linear_8bit_act_xbit_weight.h#L14), which specifies kernel function pointers for preparing activation data, preparing weight data, and running the kernel. The operator, responsible for tiling and scheduling, interacts with kernels solely through this config.


## Performance

In the table below, we show Llama3.1 8B token generation performance using 6 CPU threads on an M1 Macbook Pro with 32GB of RAM. 


<table class="table table-bordered">
  <tr>
   <td><strong>Bitwidth x</strong>
   </td>
   <td><strong>torch.compile (Decode tokens/sec)</strong>
   </td>
   <td><strong>ExecuTorch (Decode tokens/sec)</strong>
   </td>
   <td><strong>ExecuTorch PTE size (GiB)</strong>
   </td>
  </tr>
  <tr>
   <td>1
   </td>
   <td>24.18
   </td>
   <td>17.86
   </td>
   <td>1.46
   </td>
  </tr>
  <tr>
   <td>2
   </td>
   <td>27.02
   </td>
   <td>19.65
   </td>
   <td>2.46
   </td>
  </tr>
  <tr>
   <td>3
   </td>
   <td>21.01
   </td>
   <td>22.25
   </td>
   <td>3.46
   </td>
  </tr>
  <tr>
   <td>4
   </td>
   <td>19.51
   </td>
   <td>19.47
   </td>
   <td>4.47
   </td>
  </tr>
  <tr>
   <td>5
   </td>
   <td>14.78
   </td>
   <td>16.34
   </td>
   <td>5.47
   </td>
  </tr>
  <tr>
   <td>6
   </td>
   <td>12.80
   </td>
   <td>13.61
   </td>
   <td>6.47
   </td>
  </tr>
  <tr>
   <td>7
   </td>
   <td>8.16
   </td>
   <td>11.73
   </td>
   <td>7.48
   </td>
  </tr>
</table>


Results were run on an M1 Macbook Pro (with 8 perf cores, and 2 efficiency cores) with 32GB of RAM and 6 threads [using torchchat](https://github.com/pytorch/torchchat). In each test, the max-seq-length of 128 tokens were generated. For each bit width x, the embedding layer was groupwise quantized to x-bits with group size 32. In the linear layers, activations were dynamically quantized per token to 8 bits and weights were groupwise quantized to x-bits with group size 256.  Our focus here is performance and we do not report accuracy or perplexity numbers. Depending on the model, lower bit widths may require quantization-aware training, quantizing a model with a mixture of bit widths, or adjusting the group sizes for acceptable accuracy.


![Llama 3.1 chart](/assets/images/hi-po-low-bit.png){:style="width:100%"}


## Try them out and contribute!

If you want to see the new low-bit kernels in action, give them a try by [setting up torchchat](https://github.com/pytorch/torchchat/tree/main) and [quantizing and running an LLM locally using the kernels](https://github.com/pytorch/torchchat/blob/main/docs/quantization.md#experimental-torchao-lowbit-kernels).

If you want to help contribute, consider adding support for one of the following areas:

* [Add universal low-bit GEMM kernels](https://github.com/pytorch/ao/issues/1394) for Arm CPU, reusing the same bitpacking routines from the universal GEMV kernels.
* [Improve runtime selection](https://github.com/pytorch/ao/issues/1376) of ukernel configs based on ISA, packing format, and activation shape.
* Add low-bit kernels for other CPU ISAs like x86.
* Integrate third-party libraries like [KleidiAI](https://gitlab.arm.com/kleidi/kleidiai) with the operator framework.