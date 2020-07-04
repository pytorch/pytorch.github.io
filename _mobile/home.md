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

There is a growing need to execute ML models on edge devices to reduce latency, preserve privacy and enable new interactive use cases. In the past, engineers used to train models separately. They would then go through a multi-step, error prone and often complex process to transform the models for execution on a mobile device. The mobile runtime was often significantly different from the operations available during training leading to inconsistent developer and eventually user experience.

PyTorch Mobile removes these friction surfaces by allowing a seamless process to go from training to deployment by staying entirely within the PyTorch ecosystem. It provides an end-to-end workflow that simplifies the research to production environment for mobile devices. In addition, it paves the way for privacy-preserving features via Federated Learning techniques.

PyTorch Mobile is in beta stage right now and in wide scale production use. It will soon be available as a stable release once the APIs are locked down.

Key features of PyTorch Mobile:

* Available for [iOS]({{site.baseurl}}/mobile/ios), [Android]({{site.baseurl}}/mobile/android) and Linux
* Provides APIs that cover common preprocessing and integration tasks needed for incorporating ML in mobile applications
* Support for tracing and scripting via TorchScript IR
* Support for XNNPACK floating point kernel libraries for Arm CPUs
* Integration of QNNPACK for 8-bit quantized kernels. Includes support for per-channel quantization, dynamic quantization and more
* Build level optimization and selective compilation depending on the operators needed for user applications, i.e., the final binary size of the app is determined by the actual operators the app needs
* Support for hardware backends like GPU, DSP, NPU will be available soon

A typical workflow from training to mobile deployment with the optional model optimization steps is outlined in the following figure.
<div class="text-center">
  <img src="{{ site.baseurl }}/assets/images/pytorch-mobile.png" width="100%">
</div>

<!-- Do not remove the below script -->

<script page-id="home" src="{{ site.baseurl }}/assets/menu-tab-selection.js"></script>
