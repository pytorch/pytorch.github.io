---
layout: get_started
title: PyTorch for Edge
permalink: /get-started/executorch/
background-class: get-started-background
body-class: get-started
order: 5
published: true
---

## Get Started with PyTorch ExecuTorch

PyTorch’s edge specific library is [ExecuTorch](https://github.com/pytorch/executorch/)] and is designed to be lightweight, very performant even on devices with constrained hardware such as mobile phones, embedded systems and microcontrollers.

ExecuTorch relies heavily on PyTorch core technologies such as [torch.compile](https://pytorch.org/docs/stable/torch.compiler.html) and [torch.export](https://pytorch.org/docs/main/export.html), and should be very familiar to anyone who has used PyTorch in the past.

### Getting Started
You can get started by following the [general getting started guide](https://pytorch.org/executorch/stable/getting-started.html#) or jump to the specific steps for your target device.

[Using ExecuTorch on Android](https://pytorch.org/executorch/stable/using-executorch-android.html)
[Using ExecuTorch on iOS](https://pytorch.org/executorch/stable/using-executorch-ios.html)
[Using ExecuTorch with C++](https://pytorch.org/executorch/stable/using-executorch-cpp.html)

### Hardware Acceleration
ExecuTorch provides out of the box hardware acceleration for a growing number of chip manufacturers. See the following resources to learn more about how to leverage them:

* [Backend Overview](https://pytorch.org/executorch/stable/backends-overview.html)
* [XNNPACK](https://pytorch.org/executorch/stable/backends-xnnpack.html)
* [Core ML](https://pytorch.org/executorch/stable/backends-coreml.html)
* [MPS](https://pytorch.org/executorch/stable/backends-mps.html)
* [Vulkan](https://pytorch.org/executorch/stable/backends-vulkan.html)
* [ARM Ethos-U](https://pytorch.org/executorch/stable/backends-arm-ethos-u.html)
* [Qualcomm AI Engine](https://pytorch.org/executorch/stable/backends-qualcomm.html)
* [MediaTek](https://pytorch.org/executorch/stable/backends-mediatek.html)
* [Cadence Xtensa](https://pytorch.org/executorch/main/backends-cadence.html)


<script page-id="mobile" src="{{ site.baseurl }}/assets/menu-tab-selection.js"></script>
<script src="{{ site.baseurl }}/assets/get-started-sidebar.js"></script>
