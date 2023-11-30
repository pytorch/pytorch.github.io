---
layout: blog_detail
title: "PyTorch 2.1 Contains New Performance Features for AI Developers"
author: Intel
---

We are excited to see the release of PyTorch 2.1. In this blog, we discuss the five features for which Intel made significant contributions to PyTorch 2.1:

1. TorchInductor-CPU optimizations including Bfloat16 inference path for torch.compile
2. CPU dynamic shape inference path for torch.compile
3. C++ wrapper (prototype)
4. Flash-attention-based scaled dot product algorithm for CPU
5. PyTorch 2 export post-training auantization with an x86 back end through an inductor

At Intel, we are delighted to be part of the PyTorch community and appreciate the collaboration with and feedback from our colleagues at Meta* as we co-developed these features.

Let’s get started.

## TorchInductor-CPU Optimizations

This feature optimizes bfloat16 inference performance for TorchInductor. The 3rd and 4th generation Intel® Xeon® Scalable processors have built-in hardware accelerators for speeding up dot-product computation with the bfloat16 data type. Figure 1 shows a code snippet of how to specify the BF16 inference path.

```
user_model = ...

user_model.eval()
with torch.no_grad(), torch.autocast("cpu"):
	compiled_model = torch.compile(user_model)
	y = compiled_model(x)
```

Figure 1. Code snippet showing the use of BF16 inference with TorchInductor \
 


We measured the performance on three TorchInductor benchmark suites—TorchBench, Hugging Face*, and TIMM—and the results are as follows in Table 1. Here we see that performance in graph mode (TorchInductor) outperforms eager mode by factors ranging from 1.25x to 2.35x.*
 
Table 1. Bfloat16 performance geometric mean speedup in graph mode, compared with eager mode

<table class="table table-bordered">
  <tr>
   <td colspan="4" >
<strong>Bfloat16 Geometric Mean Speedup (Single-Socket Multithreads)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
torchbench
   </td>
   <td>
huggingface
   </td>
   <td>
timm_models
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.81x
   </td>
   <td>
1.25x
   </td>
   <td>
2.35x
   </td>
  </tr>
</table>



<table class="table table-bordered">
  <tr>
   <td colspan="4" >
<strong>Bfloat16 Geometric Mean Speedup (Single-Core Single Thread)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
torchbench
   </td>
   <td>
huggingface
   </td>
   <td>
timm_models
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.74x
   </td>
   <td>
1.28x
   </td>
   <td>
1.29x
   </td>
  </tr>
</table>


Developers can fully deploy their models on 4th generation Intel Xeon processors to take advantage of the Intel® Advanced Matrix Extensions (Intel® AMX) feature to get peak performance for `torch.compile`. Intel AMX has two primary components: tiles and tiled matrix multiplication (TMUL). The tiles store large amounts of data in eight two-dimensional registers, each one kilobyte in size. TMUL is an accelerator engine attached to the tiles that contain instructions to compute larger matrices in a single operation.


## CPU Dynamic Shapes Inference Path for torch.compile


Dynamic shapes is one of the key features in PyTorch 2.0. PyTorch 2.0 assumes everything is static by default. If we recompile because a size changed, we will instead attempt to recompile that size as being dynamic (sizes that have changed are likely to change in the future). Dynamic shapes support is required for popular models like large language models (LLM). Dynamic shapes that provide support for a broad scope of models can help users get more benefit from torch.compile. For dynamic shapes, we provide the post-op fusion for conv/gemm operators and vectorization code-gen for non-conv/gemm operators.

Dynamic shapes is supported by both the inductor Triton back end for CUDA* and the C++ back end for CPU. The scope covers improvements for both functionality (as measured by model passing rate) and performance (as measured by inference latency/throughput). Figure 2 shows a code snippet for the use of dynamic shape inference with TorchInductor.


```
user_model = ...

# Training example
compiled_model = torch.compile(user_model)
y = compiled_model(x_size1)
# Here trigger the recompile because the input size changed
y = compiled_model(x_size2)


# Inference example
user_model.eval()
compiled_model = torch.compile(user_model)
with torch.no_grad():
	y = compiled_model(x_size1)
 # Here trigger the recompile because the input size changed
 y = compiled_model(x_size2)
```

Figure 2. Code snippet showing the use of dynamic shape inference with TorchInductor

We again measured the performance on the three TorchInductor benchmark suites—TorchBench, Hugging Face, and TIMM—and the results are in Table 2. Here we see that performance in graph mode outperforms eager mode by factors ranging from 1.15x to 1.79x.

Table 2. Dynamic shape geometric mean speedup compared with Eager mode

<table class="table table-bordered">
  <tr>
   <td colspan="4" >
<strong>Dynamic Shape Geometric Mean Speedup (Single-Socket Multithreads)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
torchbench
   </td>
   <td>
huggingface
   </td>
   <td>
timm_models
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.35x
   </td>
   <td>
1.15x
   </td>
   <td>
1.79x
   </td>
  </tr>
</table>


<table class="table table-bordered">
  <tr>
   <td colspan="4" >
<strong>Dynamic Shape Geometric Mean Speedup (Single-Core Single-Thread)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
torchbench
   </td>
   <td>
huggingface
   </td>
   <td>
timm_models
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.48x
   </td>
   <td>
1.15x
   </td>
   <td>
1.48x
   </td>
  </tr>
</table>


## C++ Wrapper (Prototype)

The feature generates C++ code instead of Python* code to invoke the generated kernels and external kernels in TorchInductor to reduce Python overhead. It is also an intermediate step to support deployment in environments without Python.

To enable this feature, use the following configuration:


```
import torch
import torch._inductor.config as config
config.cpp_wrapper = True
```

For light workloads where the overhead of the Python wrapper is more dominant, C++ wrapper demonstrates a higher performance boost ratio. We grouped the models in TorchBench, Hugging Face, and TIMM per the average inference time of one iteration and categorized them into small, medium, and large categories. Table 3 shows the geometric mean speedups achieved by the C++ wrapper in comparison to the default Python wrapper.

Table 3. C++ wrapper geometric mean speedup compared with Eager mode

<table class="table table-bordered">
  <tr>
   <td colspan="4" >
<strong>FP32 Static Shape Mode Geometric Mean Speedup (Single-Socket Multithreads)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
Small (t &lt;= 0.04s)
   </td>
   <td>
Medium (0.04s &lt; t &lt;= 1.5s)
   </td>
   <td>
Large (t > 1.5s)
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.06x
   </td>
   <td>
1.01x
   </td>
   <td>
1.00x
   </td>
  </tr>
</table>



 


<table class="table table-bordered">
  <tr>
   <td colspan="4" >
<strong>FP32 Static Shape Mode Geometric Mean Speedup (Single-Core Single-Thread)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
Small (t &lt;= 0.04s)
   </td>
   <td>
Medium (0.04s &lt; t &lt;= 1.5s)
   </td>
   <td>
Large (t > 1.5s)
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.13x
   </td>
   <td>
1.02x
   </td>
   <td>
1.01x
   </td>
  </tr>
</table>



 


<table class="table table-bordered">
  <tr>
   <td colspan="4" >
<strong>FP32 Dynamic Shape Mode Geometric Mean Speedup (Single-Socket Multithreads)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
Small (t &lt;= 0.04s)
   </td>
   <td>
Medium (0.04s &lt; t &lt;= 1.5s)
   </td>
   <td>
Large (t > 1.5s)
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.05x
   </td>
   <td>
1.01x
   </td>
   <td>
1.00x
   </td>
  </tr>
</table>



 


<table class="table table-bordered">
  <tr>
   <td colspan="4" >
<strong>FP32 Dynamic Shape Mode Geometric Mean Speedup (Single-Core Single-Thread)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
Small (t &lt;= 0.04s)
   </td>
   <td>
Medium (0.04s &lt; t &lt;= 1.5s)
   </td>
   <td>
Large (t > 1.5s)
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.14x
   </td>
   <td>
1.02x
   </td>
   <td>
1.01x
   </td>
  </tr>
</table>



 


<table class="table table-bordered">
  <tr>
   <td colspan="4" >
<strong>BF16 Static Shape Mode Geometric Mean Speedup (Single-Socket Multithreads)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
Small (t &lt;= 0.04s)
   </td>
   <td>
Medium (0.04s &lt; t &lt;= 1.5s)
   </td>
   <td>
Large (t > 1.5s)
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.09x
   </td>
   <td>
1.03x
   </td>
   <td>
1.04x
   </td>
  </tr>
</table>



 


<table class="table table-bordered">
  <tr>
   <td colspan="4" >
<strong>BF16 Static Shape Mode Geometric Mean Speedup (Single-Core Single-Thread)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
Small (t &lt;= 0.04s)
   </td>
   <td>
Medium (0.04s &lt; t &lt;= 1.5s)
   </td>
   <td>
Large (t > 1.5s)
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.17x
   </td>
   <td>
1.04x
   </td>
   <td>
1.03x
   </td>
  </tr>
</table>


## Flash-Attention-Based Scaled Dot Product Algorithm for CPU

Scaled dot product attention (SDPA) is one of the flagship features of PyTorch 2.0 that helps speed up transformer models. It is accelerated with optimal CUDA kernels while still lacking optimized CPU kernels. This flash-attention implementation targets both training and inference, with both FP32 and Bfloat16 data types supported. There is no front-end use change for users to leverage this SDPA optimization. When calling SDPA, a specific implementation will be chosen automatically, including this new implementation.


We have measured the SDPA-related models in Hugging Face, and they are proven effective when compared to the unfused SDPA. Shown in Table 4 are the geometric mean speedups for SDPA optimization. \
 
Table 4. SDPA optimization performance geometric mean speedup

<table class="table table-bordered">
  <tr>
   <td colspan="3" >
<strong>SDPA Geometric Mean Speedup (Single-Socket Multithreads)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
Geometric Speedup FP32
   </td>
   <td>
Geometric Speedup BF16
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.15x, 20/20
   </td>
   <td>
1.07x, 20/20
   </td>
  </tr>
</table>


<table class="table table-bordered">
  <tr>
   <td colspan="3" >
<strong>SDPA Geometric Mean Speedup (Single-Core Single-Thread)</strong>
   </td>
  </tr>
  <tr>
   <td>
Compiler
   </td>
   <td>
Geometric Speedup FP32
   </td>
   <td>
Geometric Speedup BF16
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
1.02x, 20/20
   </td>
   <td>
1.04x, 20/20
   </td>
  </tr>
</table>



## PyTorch 2 Export Post-Training Quantization with x86 Back End through Inductor


PyTorch provides a new quantization flow in the PyTorch 2.0 export. This feature uses TorchInductor with an x86 CPU device as the back end for post-training static quantization with this new quantization flow. An example code snippet is shown in Figure 3.


```
import torch
import torch._dynamo as torchdynamo
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq

model = ... 

model.eval()
with torch.no_grad():
 # Step 1: Trace the model into an FX graph of flattened ATen operators
 exported_graph_module, guards = torchdynamo.export(
	 model,
	 *copy.deepcopy(example_inputs),
	 aten_graph=True,
 )

 # Step 2: Insert observers or fake quantize modules
 quantizer = xiq.X86InductorQuantizer()
 operator_config = xiq.get_default_x86_inductor_quantization_config()
 quantizer.set_global(operator_config)
 prepared_graph_module = prepare_pt2e(exported_graph_module, quantizer)

 # Doing calibration here.

 # Step 3: Quantize the model
 convert_graph_module = convert_pt2e(prepared_graph_module)

 # Step 4: Lower Quantized Model into the backend
 compile_model = torch.compile(convert_graph_module)
```

Figure 3. Code snippet showing the use of Inductor as back end for PyTorch 2 export post-training quantization

All convolutional neural networks (CNN) models from the TorchBench test suite have been measured and proven effective when compared with the Inductor FP32 inference path. Performance metrics are shown in Table 5.


<table class="table table-bordered">
  <tr>
   <td>
<strong>Compiler</strong>
   </td>
   <td>
<strong>Geometric Speedup</strong>
   </td>
   <td>
<strong>Geometric Related Accuracy Loss</strong>
   </td>
  </tr>
  <tr>
   <td>
inductor
   </td>
   <td>
3.25x, 12/12
   </td>
   <td>
0.44%, 12/12
   </td>
  </tr>
</table>


## Next Steps


### Get the Software


Try out [PyTorch 2.1](https://github.com/pytorch/pytorch/releases/tag/v2.1.0) and realize the performance benefits for yourself from these features contributed by Intel.

We encourage you to check out Intel’s other [AI Tools](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/tools.html) and [framework](https://www.intel.com/content/www/us/en/developer/tools/frameworks/overview.html) optimizations and learn about the open, standards-based [oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html) multiarchitecture, multivendor programming model that forms the foundation of Intel’s AI software portfolio.

For more details about the 4th generation Intel Xeon Scalable processor, visit the [AI platform](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/platform.html) where you can learn how Intel is empowering developers to run high-performance, efficient end-to-end AI pipelines.


### PyTorch Resources

* [PyTorch Get Started](http://pytorch.org/get-started/pytorch-2.0/)
* [Dev Discussions](http://dev-discuss.pytorch.org/t/pytorch-release-2-0-execution-update/1077)
* [Documentation](http://pytorch.org/docs/2.0/)

### Product and Performance Information

1 Amazon EC2* m7i.16xlarge: 1-node, Intel Xeon Platinum 8488C processor with 256 GB memory (1 x 256 GB DDR5 4800 MT/s), microcode 0x2b000461, hyperthreading on, turbo on, Ubuntu* 22.04.3 LTS, kernel 6.2.0-1011-aws, GCC* 11.3.0, Amazon Elastic Block Store 200 GB, BIOS Amazon EC2 1.0 10/16/2017; Software: [PyTorch 2.1.0_rc4](https://github.com/pytorch/pytorch/tree/release/2.1), [Intel® oneAPI Deep Neural Network Library (oneDNN) version 3.1.1](https://github.com/oneapi-src/oneDNN/tree/v3.1.1), [TorchBench](https://github.com/pytorch/benchmark/commit/ffbbebb9), [TorchVision](https://github.com/pytorch/vision/commit/8636bf3), [TorchText](https://github.com/pytorch/text/commit/142d029), [TorchAudio](https://github.com/pytorch/audio/commit/475b6ae), [TorchData](https://github.com/pytorch/data/commit/eb9bf61), [TorchDynamo Benchmarks](https://github.com/pytorch/pytorch/tree/release/2.1/benchmarks/dynamo), tested by Intel on 9/12/2023.


2 Amazon EC2 c6i.16xlarge: 1-node, Intel Xeon Platinum 8375C processor with 128 GB memory (1 x 128 GB DDR4 3200 MT/s), microcode 0xd0003a5, hyperthreading on, turbo on, Ubuntu 22.04.2 LTS, kernel 6.2.0-1011-aws, gcc 11.3.0, Amazon Elastic Block Store 200 GB, BIOS Amazon EC2 1.010/16/2017; Software: [PyTorch 2.1.0_rc4](https://github.com/pytorch/pytorch/tree/release/2.1), [oneDNN version 3.1.1](https://github.com/oneapi-src/oneDNN/tree/v3.1.1), [TorchBench](https://github.com/pytorch/benchmark/commit/ffbbebb9), [TorchVision](https://github.com/pytorch/vision/commit/8636bf3), [TorchText](https://github.com/pytorch/text/commit/142d029), [TorchAudio](https://github.com/pytorch/audio/commit/475b6ae), [TorchData](https://github.com/pytorch/data/commit/eb9bf61), [TorchDynamo Benchmarks](https://github.com/pytorch/pytorch/tree/release/2.1/benchmarks/dynamo), [TorchBench cpu userbenchmark](https://github.com/pytorch/benchmark/tree/chuanqiw/inductor_quant/userbenchmark/cpu), tested by Intel on 9/12/2023.
