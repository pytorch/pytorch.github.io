---
layout: blog_detail
title: "Unlocking the Latest Features in PyTorch 2.6 for Intel Platforms"
author: "the Intel PyTorch Team" 
---

[PyTorch* 2.6](https://pytorch.org/blog/pytorch2-6/) has just been released with a set of exciting new features including torch.compile compatibility with Python 3.13, new security and performance enhancements, and a change in the default parameter for torch.load. PyTorch also announced the deprecation of its official Anaconda channel.

Among the performance features are three that enhance developer productivity on Intel platforms:

1. Improved Intel GPU availability
2. FlexAttention optimization on x86 CPU for LLM 
3. FP16 on x86 CPU support for eager and Inductor modes

## Improved Intel GPU Availability

To provide developers working in artificial intelligence (AI) with better support for Intel GPUs, the PyTorch user experience on these GPUs has been enhanced. This improvement includes simplified installation steps, a Windows* release binary distribution, and expanded coverage of supported GPU models, including the latest Intel® Arc™ B-Series discrete graphics.

These new features help promote accelerated machine learning workflows within the PyTorch ecosystem, providing a consistent developer experience and support. Application developers and researchers seeking to fine-tune, perform inference, and develop with PyTorch models on [Intel® Core™ Ultra AI PCs ](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc.html) and [Intel® Arc™ discrete graphics](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc.html) will now be able to install PyTorch directly with binary releases for Windows, Linux*, and Windows Subsystem for Linux 2.

The new features include:

* Simplified Intel GPU software stack setup to enable one-click installation of the torch-xpu PIP wheels to run deep learning workloads in a ready-to-use fashion, thus eliminating the complexity of installing and activating Intel GPU development software bundles. 
* Windows binary releases for torch core, torchvision and torchaudio have been made available for Intel GPUs, expanding from [Intel® Core™ Ultra Series 2](https://www.intel.com/content/www/us/en/products/details/processors/core-ultra.html) with Intel® Arc™ Graphics and [Intel® Arc™ A-Series graphics ](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/desktop/a-series/overview.html)to the latest GPU hardware [Intel® Arc™ B-Series graphics](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/desktop/b-series/overview.html) support. 
* Further enhanced coverage of Aten operators on Intel GPUs with SYCL* kernels for smooth eager mode execution, as well as bug fixes and performance optimizations for torch.compile on Intel GPUs. 

Get a tour of new environment setup, PIP wheels installation, and examples on Intel® Client GPUs and Intel® Data Center GPU Max Series in the [Getting Started Guide](https://pytorch.org/docs/main/notes/get_start_xpu.html). 

## FlexAttention Optimization on X86 CPU for LLM

FlexAttention was first introduced in [PyTorch 2.5](https://pytorch.org/blog/pytorch2-5/), to address the need to support various Attentions or even combinations of them. This PyTorch API leverages torch.compile to generate a fused FlashAttention kernel, which eliminates extra memory allocation and achieves performance comparable to handwritten implementations. 

Previously, FlexAttention was implemented for CUDA* devices based on the Triton backend. Since PyTorch 2.6, X86 CPU support of FlexAttention was added through TorchInductor CPP backend. This new feature leverages and extends current CPP template abilities to support broad attention variants (e.g., PageAttention, which is critical for LLMs inference) based on the existing FlexAttention API, and brings optimized performance on x86 CPUs. With this feature, user can easily use FlexAttention API to compose their Attention solutions on CPU platforms and achieve good performance.

Typically, FlexAttention is utilized by popular LLM ecosystem projects, such as Hugging Face transformers and vLLM in their LLM related modeling (e.g., PagedAttention) to achieve better out-of-the-box performance. Before the official adoption happens, [this enabling PR](https://github.com/huggingface/transformers/pull/35419) in Hugging Face can help us the performance benefits that FlexAttention can bring on x86 CPU platforms. 

The graph below shows the performance comparison of PyTorch 2.6 (with this feature) and PyTorch 2.5 (without this feature) on typical Llama models. For real-time mode (Batch Size = 1), there is about 1.13x-1.42x performance improvement for next token across different input token lengths. As for best throughput under a typical SLA (P99 token latency &lt;=50ms), PyTorch 2.6 achieves more than 7.83x performance than PyTorch 2.5 as PyTorch 2.6 can run at 8 inputs (Batch Size = 8) together and still keep SLA while PyTorch 2.5 can only run 1 input, because FlexAttention based PagedAttention in PyTorch 2.6 provides more efficiency during multiple batch size scenarios.


![Figure 1. Performance comparison of PyTorch 2.6 and PyTorch 2.5 on Typical Llama Models](/assets/images/unlocking-pt-2-6-intel.png){:style="width:100%"}


**Figure 1. Performance comparison of PyTorch 2.6 and PyTorch 2.5 on Typical Llama Models**

## FP16 on X86 CPU Support for Eager and Inductor Modes

Float16 is a commonly used reduced floating-point type that improves performance in neural network inference and training. CPUs like recently launched [Intel® Xeon® 6 with P-Cores](https://www.intel.com/content/www/us/en/products/details/processors/xeon/xeon6-p-cores.html) support Float16 datatype with native accelerator [AMX](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html), which highly improves the Float16 performance. Float16 support on x86 CPU was first introduced in PyTorch 2.5 as a prototype feature. Now it has been further improved for both eager mode and Torch.compile + Inductor mode, which is pushed to Beta level for broader adoption. This helps the deployment on the CPU side without the need to modify the model weights when the model is pre-trained with mixed precision of Float16/Float32. On platforms that support AMX Float16 (i.e., the Intel® Xeon® 6 processors with P-cores), Float16 has the same pass rate as Bfloat16 across the typical PyTorch benchmark suites: TorchBench, Hugging Face, and Timms. It also shows good performance comparable to 16 bit datatype Bfloat16.

## Summary

In this blog, we discussed three features to enhance developer productivity on Intel platforms in PyTorch 2.6.  These three features are designed to improve Intel GPU availability, optimize FlexAttention for x86 CPUs tailored for large language models (LLMs), and support FP16 on x86 CPUs in both eager and Inductor modes. Get [PyTorch 2.6](https://pytorch.org/) and try them for yourself or you can access PyTorch 2.6 on the [Intel® Tiber™ AI Cloud](https://ai.cloud.intel.com/) to take advantage of hosted notebooks that are optimized for Intel hardware and software. 

## Acknowledgements

The release of PyTorch 2.6 is an exciting milestone for Intel platforms, and it would not have been possible without the deep collaboration and contributions from the community. We extend our heartfelt thanks to [Alban](https://github.com/albanD), [Andrey](https://github.com/atalman), [Bin](https://github.com/desertfire), [Jason](https://github.com/jansel), [Jerry](https://github.com/jerryzh168) and [Nikita](https://github.com/malfet) for sharing their invaluable ideas, meticulously reviewing PRs, and providing insightful feedback on RFCs. Their dedication has driven continuous improvements and pushed the ecosystem forward for Intel platforms.

## References

* [FlexAttention in PyTorch](https://pytorch.org/blog/flexattention/)
* [PagedAttention Optimization](https://arxiv.org/abs/2309.06180) 
* [Intel® Xeon® 6 with P-Cores](•%09https:/www.intel.com/content/www/us/en/products/details/processors/xeon/xeon6-p-cores.html) 

## Product and Performance Information

Measurement on AWS EC2 m7i.metal-48xl using: 2x Intel® Xeon® Platinum 8488C, HT On, Turbo On, NUMA 2, Integrated Accelerators Available [used]: DLB [8], DSA [8], IAA[8], QAT[on CPU, 8], Total Memory 512GB (16x32GB DDR5 4800 MT/s [4400 MT/s]), BIOS Amazon EC2 1.0, microcode 0x2b000603, 1x Elastic Network Adapter (ENA) 1x Amazon Elastic Block Store 800G, Ubuntu 24.04.1 LTS 6.8.0-1018-aws  Test by Intel on Jan 15<sup>th</sup> 2025.

## Notices and Disclaimers

Performance varies by use, configuration and other factors. Learn more on the Performance Index site. Performance results are based on testing as of dates shown in configurations and may not reflect all publicly available updates.  See backup for configuration details.  No product or component can be absolutely secure. Your costs and results may vary. Intel technologies may require enabled hardware, software or service activation.

Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.

## AI disclaimer:

AI features may require software purchase, subscription or enablement by a software or platform provider, or may have specific configuration or compatibility requirements. Details at [www.intel.com/AIPC](http://www.intel.com/AIPC). Results may vary.