---
layout: blog_detail
title: 'Prototype Features Now Available - APIs for Hardware Accelerated Mobile and ARM64 Builds'
author: Team PyTorch
---

Today, we are announcing four PyTorch prototype features for Mobile and ARM64 architectures to enable developers to get maximized on-device model performance.  These features include:

* Neural Networks API (NNAPI) Support in collaboration with Google Android
* GPU support on Android via Vulkan
* GPU Support on iOS via Metal
* ARM64 Builds for Linux

NNAPI and GPU support provides developers with access to the wide range of mobile hardware available directly through PyTorch Mobile: CPU and GPU on iOS and Android devices, as well as additional neural network accelerators and Digital Signal Processors (DSPs) on Android devices. Providing PyTorch builds for ARM64 on Linux will also allow developers to build directly on popular hardware leveraging these processors. 

Below, you’ll find brief descriptions of each feature with the links to get you started. These features are available through our [nightly builds](https://pytorch.org/). Reach out to us on the [PyTorch Forums](https://discuss.pytorch.org/) for any comment or feedback. We would love to get your feedback on those and hear how you are using them!

## NNAPI Support with Google Android

The Google Android Team and PyTorch worked together to provide support for Android’s Neural Networks API (NNAPI) via PyTorch Mobile. This can help developers unlock high-performance execution on Android phones by accessing all mobile hardware. NNAPI allows Android apps to run computationally intensive neural networks on the most powerful and efficient parts of the chips that power mobile phones, including GPUs (Graphics Processing Units) and NPUs (specialized Neural Processing Units). It was introduced in Android 8 (Oreo) and significantly expanded in Android 10 and 11 to support a richer set of AI models. Developers access those do this directly from PyTorch Mobile to leverage its best-in-class experience for ML developers. This initial release includes fully-functional support for a small but powerful set of features and operators, and we will be expanding support in the coming months. 

**Links**
* [Android Blog: Android Neural Networks API 1.3 and PyTorch Mobile support](https://android-developers.googleblog.com/2020/11/android-neural-networks-api-13.html)
* [PyTorch Medium Blog: Support for Android NNAPI with PyTorch Mobile](http://bit.ly/android-nnapi-pytorch-mobile-announcement)

## PyTorch Mobile GPU support

Inferencing on GPU can provide great performance on many models types and especially float models. It often allows models to run with lower battery consumption on device GPUs for SoCs like Qualcomm, MediaTek, and Apple A Series. The prototype level support we provide for on device GPUs for both iOS via Metal and Android via Vulkan is at an early stage: performance is not optimized and model coverage is limited. We expect this to improve significantly over the course of 2021 and would like to hear from you which models and devices you would like to see performance improvements on.

**Links**

* [Prototype source workflows](https://github.com/pytorch/tutorials/tree/master/prototype_source)

## ARM64 Builds for Linux

We will now provide prototype level PyTorch builds for ARM64 devices on Linux. As we see more ARM usage in our community with platforms such as Raspberry Pis and Graviton(2) instances spanning both at the edge and on servers respectively. This feature is available through our [nightly builds](https://pytorch.org/).

We value your feedback on these features and look forward to collaborating with you to continuously improve them further!

Thank you,

Team PyTorch


