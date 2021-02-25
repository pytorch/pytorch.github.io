---
layout: mobile
title: Home
permalink: /mobile/home/
background-class: mobile-background
body-class: mobile
order: 1
published: true
redirect_from: "/mobile/"
---

# PyTorch Mobile

There is a growing need to execute ML models on edge devices to reduce latency, preserve privacy, and enable new interactive use cases.

The PyTorch Mobile runtime beta release allows you to seamlessly go from training a model to deploying it, while staying entirely within the PyTorch ecosystem. It provides an end-to-end workflow that simplifies the research to production environment for mobile devices. In addition, it paves the way for privacy-preserving features via federated learning techniques.

PyTorch Mobile is in beta stage right now, and is already in wide scale production use. It will soon be available as a stable release once the APIs are locked down.


## Key features
* Available for [iOS]({{site.baseurl}}/mobile/ios), [Android]({{site.baseurl}}/mobile/android) and Linux
* Provides APIs that cover common preprocessing and integration tasks needed for incorporating ML in mobile applications
* Support for tracing and scripting via TorchScript IR
* Support for XNNPACK floating point kernel libraries for Arm CPUs
* Integration of QNNPACK for 8-bit quantized kernels. Includes support for per-channel quantization, dynamic quantization and more
* Build level optimization and selective compilation depending on the operators needed for user applications, i.e., the final binary size of the app is determined by the actual operators the app needs
* Streamline model optimization via optimize_for_mobile
* Support for hardware backends like GPU, DSP, and NPU will be available soon in Beta


## Prototypes
We have launched the following features in prototype, available in the PyTorch nightly releases, and would love to get your feedback on the [PyTorch forums](https://discuss.pytorch.org/c/mobile/18):

* Runtime binary size reduction via our Lite Interpreter
* GPU support on [iOS via Metal](https://pytorch.org/tutorials/prototype/ios_gpu_workflow.html)
* GPU support on [Android via Vulkan](https://pytorch.org/tutorials/prototype/vulkan_workflow.html)
* DSP and NPU support on Android via [Google NNAPI](https://pytorch.org/tutorials/prototype/nnapi_mobilenetv2.html)

## Models
You can now access state of the art pre-trained models:
* [MobileNetV3/V2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)
* D2Go
* [PyTorch Hub](https://pytorch.org/hub/)
* [PyTorch Tutorials](https://pytorch.org/tutorials)


## Deployment workflow

A typical workflow from training to mobile deployment with the optional model optimization steps is outlined in the following figure.
<div class="text-center">
  <img src="{{ site.baseurl }}/assets/images/pytorch-mobile.png" width="100%">
</div>

## Examples to get you started

* [PyTorch Mobile Runtime for iOS](https://www.youtube.com/watch?v=amTepUIR93k)
* [PyTorch Mobile Runtime for Android](https://www.youtube.com/watch?v=5Lxuu16_28o)
* [PyTorch Mobile Recipes in Tutorials](https://pytorch.org/tutorials/recipes/ptmobile_recipes_summary.html)


<!-- Do not remove the below script -->

<script page-id="home" src="{{ site.baseurl }}/assets/menu-tab-selection.js"></script>

<!-- Do not remove the below script -->

<script page-id="home" src="{{ site.baseurl }}/assets/menu-tab-selection.js"></script>
