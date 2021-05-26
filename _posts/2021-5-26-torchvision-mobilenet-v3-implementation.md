---
layout: blog_detail
title: 'Everything you need to know about TorchVision’s MobileNetV3 implementation'
author: Vasilis Vryniotis and Francisco Massa
---

In TorchVision v0.9, we released a series of [new mobile-friendly models](https://pytorch.org/blog/ml-models-torchvision-v0.9/)  that can be used for Classification, ObjectDetection and Semantic Segmentation. In this article, we will dig deep into the code of the models, share notableimplementation details, explain how we configured and trained them, and highlight important tradeoffs we madeduring their tuning. Our goal is to disclose technical details that typically remain undocumented in the original papersand repos of the models.

### Network Architecture

The implementation of the [MobileNetV3 architecture](https://github.com/pytorch/vision/blob/11bf27e37190b320216c349e39b085fb33aefed1/torchvision/models/mobilenetv3.py) follows closely the [original paper](https://arxiv.org/abs/1905.02244). It is customizable and offersdifferent configurations for building Classification, Object Detection and Semantic Segmentation backbones. It wasdesigned to follow a similar structure to MobileNetV2 and the two share [common building blocks](https://github.com/pytorch/vision/blob/cac8a97b0bd14eddeff56f87a890d5cc85776e18/torchvision/models/mobilenetv2.py#L32).

Off-the-shelf, we offer the two variants described on the paper: the [Large](https://github.com/pytorch/vision/blob/11bf27e37190b320216c349e39b085fb33aefed1/torchvision/models/mobilenetv3.py#L196-L214) and the [Small](https://github.com/pytorch/vision/blob/11bf27e37190b320216c349e39b085fb33aefed1/torchvision/models/mobilenetv3.py#L215-L229). Both are constructed using thesame code with the only difference being their configuration which describes the number of blocks, their sizes, theiractivation functions etc.

### Configuration parameters

Even though one can write a [custom InvertedResidual setting](https://github.com/pytorch/vision/blob/11bf27e37190b320216c349e39b085fb33aefed1/torchvision/models/mobilenetv3.py#L105) and pass it to the MobileNetV3 class directly, for the majority of applications we can adapt the existing configs by passing parameters to the [model building methods](https://github.com/pytorch/vision/blob/11bf27e37190b320216c349e39b085fb33aefed1/torchvision/models/mobilenetv3.py#L253). Some of the key configuration parameters are the following:

- The width_mult parameter is a multiplier that affects the number of channels of the model. The default value is 1 and by increasing or decreasing it one can change the number of filters of all convolutions, including the ones of the first and last layers. The implementation ensures that the number of filters is always a multiple of 8. This is a hardware optimization trick which allows for faster vectorization of operations.
- The reduced_tail parameter halves the number of channels on the last blocks of the network. This version is used by some Object Detection and Semantic Segmentation models. It’s a speed optimization which is described on the MobileNetV3 paper and reportedly leads to a 15% latency reduction without a significant negative effect on accuracy.
- The dilated parameter affects the last 3 InvertedResidual blocks of the model and turns their normal depthwise Convolutions to Atrous Convolutions. This is used to control the output stride of these blocks and has a significant positive effect on the accuracy of Semantic Segmentation models.
