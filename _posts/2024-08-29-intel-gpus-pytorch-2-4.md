---
layout: blog_detail
title: "Accelerate Your AI: PyTorch 2.4 Now Supports Intel GPUs for Faster Workloads"
author: the PyTorch Team at Intel
---

We have exciting news\! PyTorch 2.4 now supports Intel® Data Center GPU Max Series and the SYCL software stack, making it easier to speed up your AI workflows for both training and inference. This update allows for you to have a consistent programming experience with minimal coding effort and extends PyTorch’s device and runtime capabilities, including device, stream, event, generator, allocator, and guard, to seamlessly support streaming devices. This enhancement simplifies deploying PyTorch on ubiquitous hardware, making it easier for you to integrate different hardware back ends. 

Intel GPU support upstreamed into PyTorch provides support for both eager and graph modes, fully running Dynamo Hugging Face benchmarks. Eager mode now includes common Aten operators implemented with SYCL. The most performance-critical graphs and operators are highly optimized by using oneAPI Deep Neural Network Library (oneDNN) and oneAPI Math Kernel Library (oneMKL). Graph mode (torch.compile) now has an enabled Intel GPU back end to implement the optimization for Intel GPUs and to integrate Triton. Furthermore, data types such as FP32, BF16, FP16, and automatic mixed precision (AMP) are supported. The PyTorch Profiler, based on Kineto and oneMKL, is being developed for the upcoming PyTorch 2.5 release. 

Take a look at the current and planned front-end and back-end improvements for Intel GPU upstreamed into PyTorch.

![the current and planned front-end and back-end improvements for Intel GPU upstreamed into PyTorch](/assets/images/intel-gpus-pytorch-2-4.jpg){:style="width:100%"}

PyTorch 2.4 on Linux supports Intel Data Center GPU Max Series for training and inference while maintaining the same user experience as other hardware. If you’re migrating code from CUDA, you can run your existing application on an Intel GPU with minimal changes—just update the device name from `cuda` to `xpu`. For example:

```
# CUDA Code 
tensor = torch.tensor([1.0, 2.0]).to("cuda") 
 
# Code for Intel GPU 
tensor = torch.tensor([1.0, 2.0]).to("xpu")
```

## Get Started

Try PyTorch 2.4 on the Intel Data Center GPU Max Series through the [Intel® Tiber™ Developer Cloud](https://cloud.intel.com/). Get a tour of the [environment setup, source build, and examples](https://pytorch.org/docs/main/notes/get\_start\_xpu.html\#examples). To learn how to create a free Standard account, see [Get Started](https://console.cloud.intel.com/docs/guides/get\_started.html), then do the following:

1. Sign in to the [cloud console](https://console.cloud.intel.com/docs/guides/get\_started.html).

2. From the [Training](https://console.cloud.intel.com/training) section, open the **PyTorch 2.4 on Intel GPUs** notebook.

3. Ensure that the **PyTorch 2.4** kernel is selected for the notebook.

## Summary

PyTorch 2.4 introduces initial support for Intel Data Center GPU Max Series to accelerate your AI workloads. With Intel GPU, you’ll get continuous software support, unified distribution, and synchronized release schedules for a smoother development experience. We’re enhancing this functionality to reach Beta quality in PyTorch 2.5. Planned features in 2.5 include:

* More Aten operators and full Dynamo Torchbench and TIMM support in Eager Mode.

* Full Dynamo Torchbench and TIMM benchmark support in torch.compile.

* Intel GPU support in torch.profile.

* PyPI wheels distribution.

* Windows and Intel Client GPU Series support.

We welcome the community to evaluate these new contributions to [Intel GPU support on PyTorch](https://github.com/pytorch/pytorch?tab=readme-ov-file\#intel-gpu-support).  

## Resources

* [PyTorch 2.4: Get Started on an Intel GPU](https://pytorch.org/docs/main/notes/get\_start\_xpu.html) 

* [PyTorch Release Notes](https://github.com/pytorch/pytorch/releases)  

## Acknowledgments

We want thank PyTorch open source community for their technical discussions and insights: [Nikita Shulga](https://github.com/malfet), [Jason Ansel](https://github.com/jansel), [Andrey Talman](https://github.com/atalman), [Alban Desmaison](https://github.com/alband), and [Bin Bao](https://github.com/desertfire).

We also thank collaborators from PyTorch for their professional support and guidance.  

1 To enable GPU support and improve performance, we suggest installing the [Intel® Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/) 
