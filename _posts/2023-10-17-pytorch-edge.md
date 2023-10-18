---
layout: blog_detail
title: "PyTorch Edge: Enabling On-Device Inference Across Mobile and Edge Devices with ExecuTorch"
author:  the PyTorch Edge Team
---

We are excited to announce ExecuTorch, our all-new solution for enabling on-device inference capabilities across mobile and edge devices with the backing of industry leaders like Arm, Apple, and Qualcomm Innovation Center. 

As part of PyTorch Edge's vision for the future of the on-device AI stack and ecosystem, ExecuTorch addresses the fragmentation in the on-device AI ecosystem. It offers a design that provides extension points for seamless third-party integration to accelerate ML models on specialized hardware. Our partners have contributed custom delegate implementations to optimize model inference execution on their respective hardware platforms.

We have created extensive documentation that provides more details about ExecuTorch’s architecture, its high-level components, example ML models running on ExecuTorch, and end-to-end tutorials for exporting and running a model on various hardware devices. We are excited to see all of the innovative use cases of ExecuTorch built by the community.


## Key Components of ExecuTorch

ExecuTorch offers a compact runtime with a lightweight operator registry to cover the PyTorch ecosystem of models, and a streamlined path to execute PyTorch programs on edge devices. These devices range from mobile phones to embedded hardware powered by specific delegates built by our partners. In addition, ExecuTorch ships with a Software Developer Kit (SDK) and toolchain that provide an ergonomic UX for ML Developers to go from model authoring to training and device delegation in a single PyTorch workflow. This suite of tools enables ML developers to perform on-device model profiling and better ways of debugging the original PyTorch model.

ExecuTorch is architected from the ground up in a composable manner to allow ML developers to make decisions on what components to leverage as well as entry points to extend them if needed. This design provides the following benefits to the ML community: 

* **Portability**: Compatibility with a wide variety of computing platforms, from high-end mobile phones to highly constrained embedded systems and microcontrollers.
* **Productivity**: Enabling developers to use the same toolchains and SDK from PyTorch model authoring and conversion, to debugging and deployment to a wide variety of platforms, resulting in productivity gains.
* **Performance**: Providing end users with a seamless and high-performance experience due to a lightweight runtime as well as its ability to utilize full hardware capabilities, including general purpose CPUs and specialized purpose microprocessors such as NPUs and DSPs.


## PyTorch Edge: from PyTorch Mobile to ExecuTorch

Bringing research and production environments closer together is a fundamental goal of PyTorch. ML engineers increasingly use PyTorch to author and deploy machine learning models in highly dynamic and ever-evolving environments, from servers to edge devices such as mobile phones and embedded hardware. 

With the increasing adoption of AI in Augmented Reality (AR), Virtual Reality (VR), Mixed Reality (MR), Mobile, IoT and other domains, there is a growing need for an end-to-end on-device solution that is extensible, modular, and aligned with the PyTorch stack.

PyTorch Edge builds on the same fundamental principle of improving research to production by enabling the deployment of various ML models (spanning vision, speech, NLP, translation, ranking, integrity and content creation tasks) to edge devices via a low-friction development and deployment process. It provides a framework stack that spans the universe of on-device use-cases that the PyTorch community cares about. 

PyTorch Edge provides portability of core components that is required to reach a wide spectrum of devices which are characterized by differing hardware configurations, performance and efficiency. Such portability is achieved by allowing optimization that are custom developed for the target use-cases, and developer productivity via well defined entry-points, representations, and tools to tie all this together into a thriving ecosystem. 

PyTorch Edge is the future of the on-device AI stack and ecosystem for PyTorch. We are excited to see what the community builds with ExecuTorch’s on-device inference capabilities across mobile and edge devices backed by our industry partner delegates. 

[Learn more about PyTorch Edge and ExecuTorch](https://pytorch.org/executorch/stable/index.html).
