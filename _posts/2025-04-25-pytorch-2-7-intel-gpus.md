---
layout: blog_detail
title: "Accelerate PyTorch 2.7 on Intel® GPUs"
author: the Intel PyTorch Team
---

[PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/) continues to deliver significant functionality and performance enhancements on Intel® GPU architectures to streamline AI workflows. Application developers and researchers seeking to fine-tune, inference and develop PyTorch models on Intel GPUs will now have a consistent user experience across various operating systems, including Windows, Linux and Windows Subsystem for Linux (WSL2). This is made possible through improved installation, eager mode script debugging, a performance profiler, and graph model (torch.compile) deployment. As a result, developers have greater options with a unified GPU programming paradigm for both front-end and back-end development.

## Incremental improvements of Intel GPU support in PyTorch

Since PyTorch 2.4, we've made steady improvements to Intel GPU support with each release. With PyTorch 2.7, we are excited to share that we have established a solid foundation to have Intel GPU work in both graph mode (torch.compile) and eager mode on Windows and Linux. This includes a wide range of Intel GPU products, many of which you may already access. We hope these enhancements will unlock more ubiquitous hardware for your AI research and development.

* Over time, we have expanded Intel GPU Support across Windows and Linux, including these products:
    * [Intel® Arc™ A-Series Graphics](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/desktop/a-series/overview.html)
    * [Intel® Arc™ B-Series Graphics](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/desktop/b-series/overview.html)
    * [Intel® Core™ Ultra Processors with Intel Arc Graphics](https://www.intel.com/content/www/us/en/support/articles/000097599/processors.html)
    * [Intel® Core™ Ultra Mobile Processors (Series 2) with Intel Arc Graphics](https://www.intel.com/content/www/us/en/products/docs/processors/core-ultra/core-ultra-series-2-mobile-product-brief.html)
    * [Intel® Core™ Ultra Desktop Processors (Series 2) with Intel Arc Graphics](https://www.intel.com/content/www/us/en/products/docs/processors/core-ultra/core-ultra-desktop-processors-series-2-brief.html)
    * [Intel® Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series.html)
* [Simpler installation](https://pytorch.org/docs/2.7/notes/get_start_xpu.html) of torch-xpu PIP wheels and an effortless setup experience.
* High ATen operation coverage with SYCL and oneDNN for smooth eager mode support with functionality and performance.
* Notable speedups with torch.compile through default TorchInductor and Triton backend, proved by measurable performance gains with Hugging Face, TIMM, and TorchBench benchmarks.

Check out the detailed advancements in these related release blogs:[ PyTorch 2.4](https://pytorch.org/blog/intel-gpus-pytorch-2-4/),[ PyTorch 2.5](https://pytorch.org/blog/intel-gpu-support-pytorch-2-5/), and[ PyTorch 2.6](https://pytorch.org/blog/unlocking-pt-2-6-intel/).


## What's New in PyTorch 2.7

These are the features in PyTorch 2.7  that were added to help accelerate performance on Intel GPUs.



* Improve scaled dot-product attention (SDPA) inference performance with bfloat16 and float16 to accelerate attention-based models on Intel GPUs.  
With the new SDPA optimization for Intel GPUs on PyTorch 2.7, Stable Diffusion float16 inference achieved up to 3x gain over PyTorch 2.6 release on Intel® Arc™ B580 Graphics and Intel® Core™ Ultra 7 Processor 258V with Intel® Arc™ Graphics 140V on eager mode. See Figure 1 below.


![chart](/assets/images/pytorch-2-7-intel-gpus/fg1.png){:style="width:100%"}

**Figure 1. PyTorch 2.7 Stable Diffusion Performance Gains Over PyTorch 2.6**

* Enable torch.compile on Windows 11 for Intel GPUs, delivering the performance advantages over eager mode as on Linux. With this, Intel GPUs became the first accelerator to support torch.compile on Windows. Refer to[ Windows tutorial](https://pytorch.org/tutorials/prototype/inductor_windows.html) for details.  
Graph model (torch.compile) is enabled in Windows 11 for the first time across Intel GPUs, delivering the performance advantages over eager mode as on Linux by PyTorch 2.7. The latest performance data was measured on top of PyTorch Dynamo Benchmarking Suite using Intel® Arc™ B580 Graphics on Windows showcase torch.compile speedup ratio over eager mode as shown in Figure 2. Both training and inference achieved similar significant improvements.


![chart](/assets/images/pytorch-2-7-intel-gpus/fg2.png){:style="width:100%"}

**Figure 2. Torch.compile Performance Gains Over Eager Mode on Windows**



* Optimize the performance of PyTorch 2 Export Post Training Quantization (PT2E) on Intel GPU to provide full graph mode quantization pipelines with enhanced computational efficiency. Refer to[ PT2E tutorial](https://pytorch.org/tutorials/prototype/inductor_windows.html) for details.
* Enable AOTInductor and torch.export on Linux to simplify deployment workflows. Refer to[ AOTInductor tutorial](https://pytorch.org/docs/main/torch.compiler_aot_inductor.html) for details.
* Enable profiler on both Windows and Linux to facilitate model performance analysis. Refer to the[ PyTorch profiler tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#pytorch-profiler) for details.

Review the [Getting Started on Intel GPU Guide](https://pytorch.org/docs/2.7/notes/get_start_xpu.html) for a tour of the environment setup and a quick start on Intel GPUs.


## Future Work

Looking ahead, we will continue the Intel GPU upstream efforts in future PyTorch releases to:

* Attain state-of-the-art PyTorch-native performance to showcase competitive GEMM computational efficiency for torch.compile, and enhance performance for LLM models through FlexAttention and lower precision data types.
* Broaden feature compatibility by delivering distributed XCCL backend support for Intel® Data Center GPU Max Series.
* Expand accelerator support across core PyTorch ecosystem components including torchao, torchtune, and torchtitan.

Follow along in the [PyTorch Dev Discussion](https://dev-discuss.pytorch.org/t/intel-gpu-cpu-enabling-status-and-feature-plan-2025-h1-update/2913) to learn more about Intel GPU & CPU enabling status and features. As we get further along, we will create tickets on GitHub to document our progress. 


## Summary

In this blog, we reviewed the Intel GPU upstream progress starting in PyTorch 2.4 and highlighted the new features of PyTorch 2.7 that accelerate AI workload performance across various Intel GPUs. These new features, especially SDPA on Windows, achieved up to 3x inference (Stable Diffusion, float16) gain over PyTorch 2.6 release on Intel Arc B580 Graphics and Intel Core Ultra 7 Processor 258V with Intel Arc Graphics 140V. Also, torch.compile on Windows delivers similar performance advantages over eager mode on Dynamo benchmarks as on Linux.


## Acknowledgments

We want to thank the following PyTorch maintainers for their technical discussions and insights: [Nikita Shulga](https://github.com/malfet), [Jason Ansel](https://github.com/jansel), [Andrey Talman](https://github.com/atalman), [Alban Desmaison](https://github.com/alband), and [Bin Bao](https://github.com/desertfire).

We also thank collaborators from PyTorch for their professional support and guidance.

## Product and Performance Information

Measurement on Intel Core Ultra 7 258V: 2200 MHz, 8 Core(s), 8 Logical Processor(s) with Intel Arc 140V GPU (16GB), GPU memory 18.0 GB, using Intel Graphics Driver 32.0.101.6647 (WHQL Certified), Windows 11 Pro - 24H2. And Intel Core Ultra 5 245KF: 4200 MHz, 14 Core(s), 14 Logical Processor(s), Intel Arc B580 Graphics, dedicated GPU memory 12.0 GB, shared GPU memory 15.8 GB, using Intel Graphics Driver 32.0.101.6647 (WHQL Certified), Windows 11 Enterprise LTSC - 24H2. Test by Intel on Apr 8th, 2025.

## Notices and Disclaimers

Performance varies by use, configuration and other factors. Learn more on the Performance Index site. Performance results are based on testing as of dates shown in configurations and may not reflect all publicly available updates.  See backup for configuration details.  No product or component can be absolutely secure. Your costs and results may vary. Intel technologies may require enabled hardware, software or service activation.

Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.

## AI Disclaimer

AI features may require software purchase, subscription or enablement by a software or platform provider, or may have specific configuration or compatibility requirements. Details at [www.intel.com/AIPC](http://www.intel.com/AIPC). Results may vary.