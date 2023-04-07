---
layout: blog_detail
title: "Celebrate PyTorch 2.0 with New Performance Features for AI Developers"
author: Intel
---

Congratulations to the PyTorch Foundation for its release of **PyTorch 2.0**! In this blog, I discuss the four features for which Intel made significant contributions to PyTorch 2.0:

1. TorchInductor
2. GNN
3. INT8 Inference Optimization
4. oneDNN Graph API

We at Intel are delighted to be part of the PyTorch community and appreciate the collaboration with and feedback from our colleagues at [Meta](http://www.meta.com/) as we co-developed these features.


Let’s get started.


## 1. TorchInductor CPU FP32 Inference Optimized


As part of the PyTorch 2.0 compilation stack, TorchInductor CPU backend optimization brings notable performance improvements via graph compilation over the PyTorch eager mode.


The TorchInductor CPU backend is sped up by leveraging the technologies from the [Intel® Extension for PyTorch](http://github.com/intel/intel-extension-for-pytorch) for Conv/GEMM ops with post-op fusion and weight prepacking, and PyTorch ATen CPU kernels for memory-bound ops with explicit vectorization on top of OpenMP*-based thread parallelization.


With these optimizations on top of the powerful loop fusions in TorchInductor codegen, we achieved up to a **1.7x** FP32 inference performance boost over three representative deep learning benchmarks: TorchBench, HuggingFace, and timm1. Training and low-precision support are under development.


### See the Improvements


The performance improvements on various backends are tracked on this [TouchInductor CPU Performance Dashboard](http://github.com/pytorch/pytorch/issues/93531#issuecomment-1457373890).


## Improve Graph Neural Network (GNN) in PyG for Inference and Training Performance on CPU


GNN is a powerful tool to analyze graph structure data. This feature is designed to improve GNN inference and training performance on Intel® CPUs, including the new 4th Gen Intel® Xeon® Scalable processors.


PyTorch Geometric (PyG) is a very popular library built upon PyTorch to perform GNN workflows. Currently on CPU, GNN models of PyG run slowly due to the lack of GNN-related sparse matrix multiplication operations (i.e., SpMM_reduce) and the lack of several critical kernel-level optimizations (scatter/gather, etc.) tuned for GNN compute.


To address this, optimizations are provided for message passing between adjacent neural network nodes:

* **scatter_reduce:** performance hotspot in message-passing when the edge index is stored in coordinate format (COO).
* **gather:** backward computation of scatter_reduce, specially tuned for the GNN compute when the index is an expanded tensor.
* **torch.sparse.mm with reduce flag:** performance hotspot in message-passing when the edge index is stored in compressed sparse row (CSR). Supported reduce flag for: sum, mean, amax, amin.

End-to-end performance benchmark results for both inference and training on 3rd Gen Intel® Xeon® Scalable processors 8380 platform and on 4th Gen 8480+ platform are discussed in [Accelerating PyG on Intel CPUs](http://www.pyg.org/ns-newsarticle-accelerating-pyg-on-intel-cpus).


## Optimize int8 Inference with Unified Quantization Backend for x86 CPU Platforms  


The new X86 quantization backend is a combination of [FBGEMM](http://github.com/pytorch/FBGEMM) (Facebook General Matrix-Matrix Multiplication) and [oneAPI Deep Neural Network Library (oneDNN](http://spec.oneapi.io/versions/latest/elements/oneDNN/source/index.html)) backends and replaces FBGEMM as the default quantization backend for x86 platforms. The result: better end-to-end int8 inference performance than FBGEMM.


Users access the x86 quantization backend by default for x86 platforms, and the selection between different kernels is automatically done behind the scenes. The rules of selection are based on prior performance testing data done by Intel during feature development. Thus, the x86 backend replaces FBGEMM and may offer better performance, depending on the use case.


The selection rules are:

* On platforms without VNNI (e.g., Intel® Core™ i7 processors), FBGEMM is always used.
* On platforms with VNNI (e.g., 2nd-4th Gen Intel® Xeon® Scalable processors and future platforms):
    * For linear, FBGEMM is always used.
    * For convolution layers, FBGEMM is used for depth-wise convolution whose layers > 100; otherwise, oneDNN is used.

Note that as the kernels continue to evolve.


The selection rules above are subject to change to achieve better performance. Performance metrics for through-put speed-up ratios of unified x86 backend vs. pure FBGEMM are discussed in [[RFC] Unified quantization backend for x86 CPU platforms #83888](http://github.com/pytorch/pytorch/issues/83888).


## Leverage oneDNN Graph API to Accelerate Inference on CPU 


[oneDNN Graph API](http://spec.oneapi.io/onednn-graph/latest/introduction.html) extends [oneDNN](http://spec.oneapi.io/versions/latest/elements/oneDNN/source/index.html) with a flexible graph API to maximize the optimization opportunity for generating efficient code on Intel® AI hardware. It automatically identifies the graph partitions to be accelerated via fusion. The [fusion patterns](http://github.com/oneapi-src/oneDNN/blob/dev-graph/doc/programming_model/ops_and_patterns.md#fusion-patterns) focus on fusing compute-intensive operations such as convolution, matmul, and their neighbor operations for both inference and training use cases.


Currently, BFloat16 and Float32 datatypes are supported and only inference workloads can be optimized.  BF16 is only optimized on machines with Intel® Advanced Vector Extensions 512 (Intel® AVX-512) BF16 support.


Few or no modifications are needed in PyTorch to support newer oneDNN Graph fusions/optimized kernels. To use oneDNN Graph, users can:

* Either use the API _torch.jit.enable_onednn_fusion(True)_ before JIT tracing a model, OR …
* Use its context manager, viz. _with torch.jit.fuser(“fuser3”)._
* For accelerating [BFloat16 inference](http://github.com/pytorch/pytorch/tree/master/torch/csrc/jit/codegen/onednn#example-with-bfloat16), we rely on eager-mode AMP (Automatic Mixed Precision) support in PyTorch and disable JIT mode’s AMP.

See the [PyTorch performance tuning guide](http://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-onednn-graph-with-torchscript-for-inference).


## Next Steps


### Get the Software


[Try out PyTorch 2.0](http://pytorch.org/get-started/locally/) and realize the performance benefits for yourself from these Intel-contributed features.


We encourage you to check out Intel’s other [AI Tools](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/tools.html) and [Framework](https://www.intel.com/content/www/us/en/developer/tools/frameworks/overview.html) optimizations and learn about the open, standards-based [oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html) multiarchitecture, multivendor programming model that forms the foundation of Intel’s AI software portfolio.


For more details about 4th Gen Intel Xeon Scalable processor, visit [AI Platform](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/platform.html) where you can learn about how Intel is empowering developers to run high-performance, efficient end-to-end AI pipelines.


### PyTorch Resources

* [PyTorch Get Started](http://pytorch.org/get-started/pytorch-2.0/)
* [Dev Discussions](http://dev-discuss.pytorch.org/t/pytorch-release-2-0-execution-update/1077)
* [Documentation](http://pytorch.org/docs/2.0/)
