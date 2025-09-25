---
layout: blog_detail
title: "The Path to Achieve PyTorch Performance Boost on Windows CPU"
author: Intel Corporation 
---

The challenge of PyTorch's lower CPU performance on Windows compared to Linux has been a significant issue. There are multiple factors leading to this performance disparity. Through our investigation, we've identified one of the primary reasons for poor CPU performance on Windows, which is linked to the Windows default malloc memory allocator.

In version 2.0, PyTorch on Windows with CPU directly utilizes the default malloc mechanism of Windows, when it is compared to the malloc used in PyTorch Linux version 2.0, it significantly increases the time for memory allocation, which results in decreased performance. We replaced the original Windows malloc mechanism, which PyTorch automatically calls, with another well-known malloc library developed by Microsoft, known as mimalloc. This replacement of malloc has already been released with PyTorch v2.1 and can significantly improve PyTorch's performance on Windows CPUs as shown below in Figure 1.

![Windows PC Performance Improvement](/assets/images/2024-05-21-perfboost-windows-cpu/windows_compare.png){:style="width:100%;"}

_Figure 1: Relative throughput improvement achieved by upgrading from Windows PyTorch version 2.0 to 2.1 (higher is better)._ 

From this graph, we see that PyTorch 2.1 on Windows CPU shows significant performance improvements. The variations in performance enhancements across different workloads mainly stem from varying proportions of different operations within distinct models, consequently affecting the frequency of memory access operations. It shows a comparatively smaller enhancement in BERT model performance, while there is a more substantial improvement in ResNet50 and MobileNet-v3 Large model performances.

On a high-performance CPU, memory allocation becomes a performance bottleneck. This is also why addressing this issue has led to such significant performance improvements. 

As shown in the graphs below, we see that PyTorch's performance on Windows CPUs can significantly be improved. However, there is still a noticeable gap when compared to its performance on Linux. This can be attributed to several factors, including the fact that malloc has not yet fully reached the performance level of Linux, among other reasons. Intel engineers will continue to collaborate with Meta engineers, to reduce the performance gap of PyTorch between Windows and Linux.


![Windows vs Linux Performance on PyTorch 2.0](/assets/images/2024-05-21-perfboost-windows-cpu/pytorch_20_win_linux.png){:style="width:100%;"}

_Figure 2.1: Relative performance of Windows vs Linux with PyTorch version 2.0 (higher is better)._ 

![Windows vs Linux Performance on PyTorch 2.1](/assets/images/2024-05-21-perfboost-windows-cpu/pytorch_21_win_linux.png){:style="width:100%;"}

_Figure 2.2: Relative performance of Windows vs Linux with PyTorch version 2.1 (higher is better)._ 


## HOW TO TAKE ADVANTAGE OF THE OPTIMIZATIONS

Install PyTorch version 2.1 or higher on Windows CPU from the [official repository](https://pytorch.org/get-started/locally/), and you may automatically experience a performance boost.


## CONCLUSION

When comparing PyTorch 2.0 and PyTorch 2.1, we observed varying degrees of performance improvement on Windows CPU. The extent of performance improvement becomes more pronounced as the number of memory allocation operations called within a workload increases. A more powerful CPU computing capability will also make this performance enhancement more pronounced, as the proportion of operations outside of computation increases.

To a certain extent, this performance enhancement helps to bridge the PyTorch CPU performance gap between Windows and Linux. Intel will continue to collaborate with Meta, enhance the performance of PyTorch on CPUs.

## ACKNOWLEDGMENTS

The results presented in this blog post was achieved through the collaborative effort of the Intel PyTorch team and Meta. We would like to express our sincere gratitude to [Xu Han](https://github.com/xuhancn), [Jiong Gong](https://github.com/jgong5), [Mingfei Ma](https://github.com/mingfeima), [Haozhe Zhu](https://github.com/zhuhaozhe), [Chuanqi Wang](https://github.com/chuanqi129), [Guobing Chen](https://github.com/Guobing-Chen) and [Eikan Wang](https://github.com/EikanWang). Their expertise and dedication have been instrumental in achieving the optimizations and performance improvements discussed here. Thanks to [Jiachen Pu](https://github.com/peterjc123) from community for his participation in the issue discussion and suggesting the use of [mimalloc](https://github.com/microsoft/mimalloc). We'd also like to express our gratitude to Microsoft for providing such an easily integrated and performant mallocation library.  Finally we want to thank [Jing Xu](https://github.com/jingxu10), [Weizhuo Zhang](https://github.com/WeizhuoZhang-intel) and [Zhaoqiong Zheng](https://github.com/ZhaoqiongZ) for their contributions to this blog.


### Product and Performance Information

The configurations in the table are collected with [svr-info](https://github.com/intel/svr-info). Test by Intel on April 15, 2024.


| Specification               | Configuration1                          | Configuration2                         |
|-----------------------------|----------------------------------------|----------------------------------------|
| Name                        | ThinkBook 14 G5+ IRH                   | ThinkBook 14 G5+ IRH                   |
| Time                        | Mon Apr 15 01:13:48 PM UTC 2024        | Mon Apr 15 01:13:48 PM UTC 2024        |
| System                      | LENOVO                                 | LENOVO                                 |
| Baseboard                   | LENOVO                                 | LENOVO                                 |
| Chassis                     | LENOVO                                 | LENOVO                                 |
| CPU Model                   | 13th Gen Intel(R) Core(TM) i7-13700H   | 13th Gen Intel(R) Core(TM) i7-13700H   |
| Microarchitecture           | Unknown Intel                          | Unknown Intel                          |
| Sockets                     | 1                                      | 1                                      |
| Cores per Socket            | 14                                     | 14                                     |
| Hyperthreading              | Enabled                                | Enabled                                |
| CPUs                        | 20                                     | 20                                     |
| Intel Turbo Boost           | Enabled                                | Enabled                                |
| Base Frequency              | 2.4GHz                                 | 2.4GHz                                 |
| All-core Maximum Frequency  | 4.7GHz                                 | 4.7GHz                                 |
| Maximum Frequency           | 4.8GHz                                 | 4.8GHz                                 |
| NUMA Nodes                  | 1                                      | 1                                      |
| Prefetchers                 | L2 HW: Enabled, L2 Adj.: Enabled, DCU HW: Enabled, DCU IP: Enabled | L2 HW: Enabled, L2 Adj.: Enabled, DCU HW: Enabled, DCU IP: Enabled |
| PPINs                       | -                                      | -                                      |
| Accelerators                | DLB, DSA, IAA, QAT                     | DLB, DSA, IAA, QAT                     |
| Installed Memory            | 32GB (8x4GB LPDDR4 7400 MT/s [5200 MT/s]) | 32GB (8x4GB LPDDR4 7400 MT/s [5200 MT/s]) |
| Hugepagesize                | 2048kb                                 | 2048kb                                 |
| Transparent Huge Pages      | madvise                                | madvise                                |
| Automatic NUMA Balancing    | Disabled                               | Disabled                               |
| NIC                         | "1. Raptor Lake PCH CNVi WiFi 2. Intel Corporation" | "1. Raptor Lake PCH CNVi WiFi 2. Intel Corporation" |
| Disk                        | Micron MTFDKBA512TFH 500G              | Micron MTFDKBA512TFH 500G              |
| BIOS                        | LBCN19WW                               | LBCN19WW                               |
| Microcode                   | 0x411c                                 | 0x411c                                 |
| OS                          | Windows 11 Desktop                     | Ubuntu 23.10                           |
| Kernel                      | OS Build 19045.4412                    | 6.5.0-27-generic                       |
| TDP                         | 200 watts                              | 200 watts                              |
| Power & Perf Policy         | Normal Powersave (7)                   | Normal Powersave (7)                   |
| Frequency Governor          | powersave                              | powersave                              |
| Frequency Driver            | intel_pstate                           | intel_pstate                           |
| Max C-State                 | 9                                      | 9                                      |



## Notices and Disclaimers

Performance varies by use, configuration and other factors. Learn more on the [Performance Index site](https://edc.intel.com/content/www/us/en/products/performance/benchmarks/overview/). 

Performance results are based on testing as of dates shown in [configurations](#product-and-performance-information) and may not reflect all publicly available updates.  See backup for configuration details. No product or component can be absolutely secure. Your costs and results may vary. Intel technologies may require enabled hardware, software or service activation. 

Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.

