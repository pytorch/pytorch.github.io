---
layout: blog_detail
title: 'Prototype Features Now Available - APIs for Hardware Accelerated Mobile and ARM64 Builds'
author: Team PyTorch
---

Today, we are announcing four PyTorch prototype features. The first three of these will enable Mobile machine-learning developers to execute models on the full set of hardware (HW) engines making up a system-on-chip (SOC). This gives developers options to optimize their model execution for unique performance, power, and system-level concurrency.

These features include enabling execution on the following on-device HW engines:
* DSP and NPUs using the Android Neural Networks API (NNAPI), developed in collaboration with Google
* GPU execution on Android via Vulkan
* GPU execution on iOS via Metal

This release also includes developer efficiency benefits with newly introduced support for ARM64 builds for Linux.

Below, you’ll find brief descriptions of each feature with the links to get you started. These features are available through our [nightly builds](https://pytorch.org/). Reach out to us on the [PyTorch Forums](https://discuss.pytorch.org/) for any comment or feedback. We would love to get your feedback on those and hear how you are using them!

## NNAPI Support with Google Android

The Google Android and PyTorch teams collaborated to enable support for Android’s Neural Networks API (NNAPI) via PyTorch Mobile. Developers can now unlock high-performance execution on Android phones as their machine-learning models will be able to access additional hardware blocks on the phone’s system-on-chip. NNAPI allows Android apps to run computationally intensive neural networks on the most powerful and efficient parts of the chips that power mobile phones, including DSPs (Digital Signal Processors) and NPUs (specialized Neural Processing Units). The API was introduced in Android 8 (Oreo) and significantly expanded in Android 10 and 11 to support a richer set of AI models. With this integration, developers can now seamlessly access NNAPI directly from PyTorch Mobile. This initial release includes fully-functional support for a core set of features and operators, and Google and Facebook will be working to expand capabilities in the coming months.

**Links**
* [Android Blog: Android Neural Networks API 1.3 and PyTorch Mobile support](https://android-developers.googleblog.com/2020/11/android-neural-networks-api-13.html)
* [PyTorch Medium Blog: Support for Android NNAPI with PyTorch Mobile](http://bit.ly/android-nnapi-pytorch-mobile-announcement)

## PyTorch Mobile GPU support

Inferencing on GPU can provide great performance on many models types, especially those utilizing high-precision floating-point math. Leveraging the GPU for ML model execution as those found in SOCs from Qualcomm, Mediatek, and Apple allows for CPU-offload, freeing up the Mobile CPU for non-ML use cases. This initial prototype level support provided for on device GPUs is via the Metal API specification for iOS, and the Vulkan API specification for Android. As this feature is in an early stage: performance is not optimized and model coverage is limited. We expect this to improve significantly over the course of 2021 and would like to hear from you which models and devices you would like to see performance improvements on.

**Links**
* [Prototype source workflows](https://github.com/pytorch/tutorials/tree/master/prototype_source)

## ARM64 Builds for Linux

We will now provide prototype level PyTorch builds for ARM64 devices on Linux. As we see more ARM usage in our community with platforms such as Raspberry Pis and Graviton(2) instances spanning both at the edge and on servers respectively. This feature is available through our [nightly builds](https://pytorch.org/).

We value your feedback on these features and look forward to collaborating with you to continuously improve them further!

Thank you,

Team PyTorch
