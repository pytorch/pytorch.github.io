---
layout: blog_detail
title: 'FX based Feature Extraction in TorchVision'
author: 
featured-img: 'assets/images/.png'
---

# Introduction

[FX](https://pytorch.org/docs/stable/fx.html) based feature extraction is a new [TorchVision utility](https://pytorch.org/vision/stable/feature_extraction.html) that lets us access intermediate transformations of an input during the forward pass of a PyTorch Module. It does so by symbolically tracing the forward method to produce a graph where each node represents a single operation. Nodes are named in a human-readable manner such that one may easily specify which nodes they want to access.

Did that all sound a little complicated? Not to worry as there’s a little in this article for everyone. Whether you’re a beginner or an advanced deep-vision practitioner, chances are you will want to know about FX feature extraction. If you still want more background on feature extraction in general, read on. If you’re already comfortable with that and want to know how to do it in PyTorch, skim ahead to [Existing Methods in PyTorch: Pros and Cons](https://docs.google.com/document/d/1WrwGSEVG4WePTsgjomQbUpXEXmS62bIveRwvSfCQLRg/edit#heading=h.qqct7s9qblzd). And if you already know about the challenges of doing feature extraction in PyTorch, feel free to skim forward to [FX to The Rescue](https://docs.google.com/document/d/1WrwGSEVG4WePTsgjomQbUpXEXmS62bIveRwvSfCQLRg/edit#heading=h.q32r4tje9ewf).

# A Recap on Feature Extraction

We’re all used to the idea of having a deep neural network (DNN) that takes inputs and produces outputs, and we don’t necessarily think of what happens in between. Let’s just consider a ResNet-50 classification model as an example:

<p align="center">
<img src="{{ site.url }}/assets/images/fx-figure-1.png" width="100%">
<br>
	Figure 1: ResNet-50 takes an image of a bird and transforms that into the abstract concept "bird". Source: Bird image from ImageNet.
</p>

We know though, that there are many sequential “layers” within the ResNet-50 architecture that transform the input step-by-step. In Figure 2 below, we peek under the hood to show the layers within ResNet-50, and we also show the intermediate transformations of the input as it passes through those layers.

<p align="center">
<img src="{{ site.url }}/assets/images/fx-figure-2.png" width="100%">
<br>
	Figure 2: ResNet-50 transforms the input image in multiple steps. Conceptually, we may access the intermediate transformation of the image after each one of these steps. 
Source: Bird image from ImageNet.
</p>

If we access one of those intermediate transformations we might agree to call the act of doing so: “feature extraction”. There are a variety of reasons we might choose to do so. Just to enumerate a few (for the more general case, not just ResNet-50):

* **Debugging**: We’re not sure that the components of our DNN are doing what we expect them to do, so we want to check outputs of specific sub-components.
* **Interpretability**: We want to understand how the DNN transforms the main inputs into the main outputs. Figure 2 above shows how ResNet-50 tends to gradually transform an input from a concrete to an abstract representation. Another example would be interpreting attention maps for transformer based vision models.
* **Extracting embeddings aka descriptors**: We can take one or more of the intermediate representations and combine them to produce a descriptor of the input. This can be applied as a means of comparing instances from the input domain. Think copy detection, image retrieval, or facial recognition.
* **Using a base architecture as part of a larger model for other tasks**: A typical example is to use ResNet-50 as a backbone for a feature pyramid network, passing the intermediate outputs to object detection and segmentation heads.

