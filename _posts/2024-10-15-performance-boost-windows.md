---
layout: blog_detail
title: "The Path to Achieve PyTorch Performance Boost on Windows CPU"
author: Intel Corporation
---

The challenge of PyTorch’s lower CPU performance on Windows compared to Linux has been a significant issue. There are multiple factors leading to this performance disparity. Through our investigation, we’ve identified several reasons for poor CPU performance on Windows, two primary issues have been pinpointed: the inefficiency of the Windows default malloc memory allocator and the absence of [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) for vectorization optimizations on the Windows platform. In this article, we show how PyTorch CPU performance on Windows has improved from the previous releases and where it stands as of PyTorch 2.4.1.


## Memory Allocation Optimization in PyTorch 2.1.2 and later

In versions prior to PyTorch 2.1.2, PyTorch relied on the operating system’s default malloc function for memory allocation. The default malloc memory allocation on the Windows platform was less efficient compared to the malloc implementation mechanism on the Linux platform, leading to increased memory allocation times and reduced performance. To address this, we have substituted the default Windows malloc with mimalloc, a more efficient memory allocator developed by Microsoft. This update, included with the release of PyTorch 2.1.2 and later, has significantly enhanced the CPU performance of PyTorch on Windows, as shown in Figure 1.1.



![performance comparison chart](/assets/images/performance-boost-windows/fg1.png){:style="width:100%"}


*PyTorch CPU Performance Improvement on Windows with Memory Allocation Optimization*

*Figure 1.1: Relative throughput improvement achieved by upgrading from Windows PyTorch version 2.0.1 to 2.1.2 (higher is better).*

The graph illustrates that with the release of PyTorch 2.1.2, there has been a notable enhancement in CPU performance on the Windows platform. The degree of improvement varies across different models, which can be attributed to the diverse mix of operations they perform and their corresponding memory access patterns. While the BERT model shows a modest performance gain, models like ResNet50 and MobileNet-v3 Large benefit from more pronounced improvements.

On a high-performance CPU, memory allocation becomes a performance bottleneck. This is also why addressing this issue has led to such significant performance improvements.

As shown in the graphs below, we see that PyTorch CPU performance on Windows can significantly be improved. However, there is still a noticeable gap when compared to its performance on Linux. The absence of vectorization optimizations in the Windows variant of PyTorch CPU is a key factor to the remaining performance gap.


![performance comparison chart](/assets/images/performance-boost-windows/fg2.png){:style="width:100%"}


*Windows vs Linux Performance on PyTorch 2.0.1*

*Figure 1.2: Relative performance of Windows vs Linux with PyTorch version 2.0.1 (higher is better).*


![performance comparison chart](/assets/images/performance-boost-windows/fg3.png){:style="width:100%; margin-top: 50px;"}


*Windows vs Linux Performance on PyTorch 2.1.2*

*Figure 1.3: Relative performance of Windows vs Linux with PyTorch version 2.1.2 (higher is better).*


## Vectorization Optimization in PyTorch 2.4.1 and later

Prior to PyTorch 2.4.1, the Windows build of PyTorch lacked [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) for vectorization optimizations, a feature that the Linux build leveraged for improved performance. This discrepancy was due to the [SLEEF](https://github.com/shibatch/sleef) Library’s integration issues on Windows, which is a SIMD Library for Evaluating Elementary Functions, vectorized libm and DFT and is essential for efficient trigonometric calculations. Through a collaborative effort with engineers from ARM and Qualcomm, these challenges were resolved, enabling the integration of SIMD into PyTorch for Windows. The PyTorch 2.4.1 update has thus significantly enhanced PyTorch’s CPU performance on Windows, as shown in Figure 2.1.


![performance comparison chart](/assets/images/performance-boost-windows/fg4.png){:style="width:100%"}


*PyTorch CPU Performance Improvement on Windows with Vertorization Optimization*

*Figure 2.1: Relative throughput improvement achieved by upgrading from PyTorch CPU version 2.1.2 to 2.4.1 (higher is better).*

As shown in the graph below, we see that PyTorch CPU performance on Windows ahieved the performance on Linux.


![performance comparison chart](/assets/images/performance-boost-windows/fg5.png){:style="width:100%"}


*Windows vs Linux Performance on PyTorch 2.4.1*

*Figure 2.2: Relative performance of Windows vs Linux with PyTorch version 2.4.1 (higher is better).*


## CONCLUSION

From PyTorch 2.0.1 to PyTorch 2.4.1, the CPU performance gap between Windows and Linux has been continuously narrowing. We compared the ratio of CPU performance on Windows to CPU performance on Linux across different versions, and the results are shown in the following graph.


![performance comparison chart](/assets/images/performance-boost-windows/fg6.png){:style="width:100%"}


*Windows vs Linux Performance on different version of PyTorch*

*Figure 3: Performance Ratio for Windows to Linux with different version of PyTorch (higher is better).*

The graph shows that with PyTorch 2.4.1, CPU performance on Windows has nearly converged with that on Linux, and on some models, it has even surpassed Linux. For example, in the case of DistillBERT and RoBERTa models, the CPU performance ratio of Windows to Linux has achieved a remarkable 102%. However, certain models, including MobileNet-v3, still show a performance discrepancy. Intel engineers will continue to collaborate with Meta engineers, to reduce the performance gap of PyTorch CPU between Windows and Linux.


## HOW TO TAKE ADVANTAGE OF THE OPTIMIZATIONS

Install PyTorch CPU 2.4.1 or later on Windows from the [official repository](https://pytorch.org/get-started/locally/), and you may automatically experience a performance boost with memory allocation and vectorizations.


## ACKNOWLEDGMENTS

The results presented in this blog post was achieved through the collaborative effort of the Intel PyTorch team and Meta. We would like to express our sincere gratitude to [Xu Han](https://github.com/xuhancn), [Jiong Gong](https://github.com/jgong5), [Haozhe Zhu](https://github.com/zhuhaozhe), [Mingfei Ma](https://github.com/mingfeima), [Chuanqi Wang](https://github.com/chuanqi129), [Guobing Chen](https://github.com/Guobing-Chen) and [Eikan Wang](https://github.com/EikanWang). Their expertise and dedication have been instrumental in achieving the optimizations and performance improvements discussed here. Thanks to [Jiachen Pu](https://github.com/peterjc123) from community for his participation in the issue discussion and suggesting the use of [mimalloc](https://github.com/microsoft/mimalloc). We’d also like to express our gratitude to Microsoft for providing such an easily integrated and performant mallocation library. Thanks to [Pierre Blanchard](https://github.com/blapie) , [Nathan Sircombe](https://github.com/nSircombe) from ARM and [Alex Reinking](https://github.com/alexreinking) from Adobe for their contribution in overcome the compatibility issues with the [sleef](https://github.com/shibatch/sleef) integrated to PyTorch Windows. Finally we want to thank [Jing Xu](https://github.com/jingxu10), [Weizhuo Zhang](https://github.com/WeizhuoZhang-intel) and [Zhaoqiong Zheng](https://github.com/ZhaoqiongZ) for their contributions to this blog.


### Product and Performance Information

The configurations in the table are collected with [svr-info](https://github.com/intel/svr-info). Test by Intel on August 30, 2024.


<table class="table table-bordered">
  <tr>
   <td><strong>Specification</strong>
   </td>
   <td><strong>Configuration1</strong>
   </td>
   <td><strong>Configuration2</strong>
   </td>
  </tr>
  <tr>
   <td>Name
   </td>
   <td>ThinkBook 14 G5+ IRH
   </td>
   <td>ThinkBook 14 G5+ IRH
   </td>
  </tr>
  <tr>
   <td>Time
   </td>
   <td>Fri Aug 30 02:43:02 PM UTC 2024
   </td>
   <td>Fri Aug 30 02:43:02 PM UTC 2024
   </td>
  </tr>
  <tr>
   <td>System
   </td>
   <td>LENOVO
   </td>
   <td>LENOVO
   </td>
  </tr>
  <tr>
   <td>Baseboard
   </td>
   <td>LENOVO
   </td>
   <td>LENOVO
   </td>
  </tr>
  <tr>
   <td>Chassis
   </td>
   <td>LENOVO
   </td>
   <td>LENOVO
   </td>
  </tr>
  <tr>
   <td>CPU Model
   </td>
   <td>13th Gen Intel(R) Core(TM) i7-13700H
   </td>
   <td>13th Gen Intel(R) Core(TM) i7-13700H
   </td>
  </tr>
  <tr>
   <td>Microarchitecture
   </td>
   <td>Unknown Intel
   </td>
   <td>Unknown Intel
   </td>
  </tr>
  <tr>
   <td>Sockets
   </td>
   <td>1
   </td>
   <td>1
   </td>
  </tr>
  <tr>
   <td>Cores per Socket
   </td>
   <td>14
   </td>
   <td>14
   </td>
  </tr>
  <tr>
   <td>Hyperthreading
   </td>
   <td>Enabled
   </td>
   <td>Enabled
   </td>
  </tr>
  <tr>
   <td>CPUs
   </td>
   <td>20
   </td>
   <td>20
   </td>
  </tr>
  <tr>
   <td>Intel Turbo Boost
   </td>
   <td>Enabled
   </td>
   <td>Enabled
   </td>
  </tr>
  <tr>
   <td>Base Frequency
   </td>
   <td>2.4GHz
   </td>
   <td>2.4GHz
   </td>
  </tr>
  <tr>
   <td>All-core Maximum Frequency
   </td>
   <td>4.7GHz
   </td>
   <td>4.7GHz
   </td>
  </tr>
  <tr>
   <td>Maximum Frequency
   </td>
   <td>4.8GHz
   </td>
   <td>4.8GHz
   </td>
  </tr>
  <tr>
   <td>NUMA Nodes
   </td>
   <td>1
   </td>
   <td>1
   </td>
  </tr>
  <tr>
   <td>Prefetchers
   </td>
   <td>L2 HW: Enabled, L2 Adj.: Enabled, DCU HW: Enabled, DCU IP: Enabled
   </td>
   <td>L2 HW: Enabled, L2 Adj.: Enabled, DCU HW: Enabled, DCU IP: Enabled
   </td>
  </tr>
  <tr>
   <td>PPINs
   </td>
   <td>-
   </td>
   <td>-
   </td>
  </tr>
  <tr>
   <td>Accelerators
   </td>
   <td>DLB, DSA, IAA, QAT
   </td>
   <td>DLB, DSA, IAA, QAT
   </td>
  </tr>
  <tr>
   <td>Installed Memory
   </td>
   <td>32GB (8x4GB LPDDR4 7400 MT/s [5200 MT/s])
   </td>
   <td>32GB (8x4GB LPDDR4 7400 MT/s [5200 MT/s])
   </td>
  </tr>
  <tr>
   <td>Hugepagesize
   </td>
   <td>2048kb
   </td>
   <td>2048kb
   </td>
  </tr>
  <tr>
   <td>Transparent Huge Pages
   </td>
   <td>madvise
   </td>
   <td>madvise
   </td>
  </tr>
  <tr>
   <td>Automatic NUMA Balancing
   </td>
   <td>Disabled
   </td>
   <td>Disabled
   </td>
  </tr>
  <tr>
   <td>NIC
   </td>
   <td>“1. Raptor Lake PCH CNVi WiFi 2. Intel Corporation”
   </td>
   <td>“1. Raptor Lake PCH CNVi WiFi 2. Intel Corporation”
   </td>
  </tr>
  <tr>
   <td>Disk
   </td>
   <td>Micron MTFDKBA512TFH 500G
   </td>
   <td>Micron MTFDKBA512TFH 500G
   </td>
  </tr>
  <tr>
   <td>BIOS
   </td>
   <td>LBCN22WW
   </td>
   <td>LBCN22WW
   </td>
  </tr>
  <tr>
   <td>Microcode
   </td>
   <td>0x411c
   </td>
   <td>0x411c
   </td>
  </tr>
  <tr>
   <td>OS
   </td>
   <td>Windows 11 Desktop
   </td>
   <td>Ubuntu 23.10
   </td>
  </tr>
  <tr>
   <td>Kernel
   </td>
   <td>OS Build 19045.4412
   </td>
   <td>6.5.0-27-generic
   </td>
  </tr>
  <tr>
   <td>TDP
   </td>
   <td>200 watts
   </td>
   <td>200 watts
   </td>
  </tr>
  <tr>
   <td>Power & Perf Policy
   </td>
   <td>Normal Powersave (7)
   </td>
   <td>Normal Powersave (7)
   </td>
  </tr>
  <tr>
   <td>Frequency Governor
   </td>
   <td>performance
   </td>
   <td>performance
   </td>
  </tr>
  <tr>
   <td>Frequency Driver
   </td>
   <td>intel_pstate
   </td>
   <td>intel_pstate
   </td>
  </tr>
  <tr>
   <td>Max C-State
   </td>
   <td>9
   </td>
   <td>9
   </td>
  </tr>
</table>



## Notices and Disclaimers

Performance varies by use, configuration and other factors. Learn more on the [Performance Index site](https://edc.intel.com/content/www/us/en/products/performance/benchmarks/overview/).

Performance results are based on testing as of dates shown in [configurations](#product-and-performance-information) and may not reflect all publicly available updates. See backup for configuration details. No product or component can be absolutely secure. Your costs and results may vary. Intel technologies may require enabled hardware, software or service activation.

Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.
