---
layout: blog_detail
title: "Extending TorchVision’s Transforms to Object Detection, Segmentation & Video tasks"
author: Philip Meier, Victor Fomin, Vasilis Vryniotis
featured-img: ""
---

TorchVision is extending its Transforms API! Here is what’s new:

- You can use them not only for Image Classification but also for Object Detection, Instance & Semantic Segmentation and Video Classification.
- You can import directly from TorchVision several SoTA data-augmentations such as MixUp, CutMix, Large Scale Jitter and SimpleCopyPaste.  
- You can use new functional transforms for transforming Videos, Bounding Boxes and Segmentation Masks.

The interface remains the same to assist the migration and adoption. The new API is currently in Prototype and we would love to get early feedback from you to improve its functionality. Please [reach out to us](https://github.com/pytorch/vision/issues/6753) if you have any questions or suggestions.

## Limitations of current Transforms

The stable Transforms API of TorchVision (aka V1) only supports single images. As a result it can only be used for classification tasks:

<p align="center">
  <img src="" width="90%">
</p>

The above approach doesn’t support Object Detection, Segmentation or Classification transforms that require the use of Labels (such as MixUp & CutMix). This limitation made any non-classification Computer Vision tasks second-class citizens as one couldn’t use the Transforms API to perform the necessary augmentations. Historically this made it difficult to train high-accuracy models using TorchVision’s primitives and thus our Model Zoo lagged by several points from SoTA.

To circumvent this limitation, TorchVision offered [custom implementations](https://github.com/pytorch/vision/blob/main/references/detection/transforms.py) in its reference scripts that show-cased how one could perform augmentations in each task. Though this practice enabled us to train high accuracy [classification](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/), [object detection & segmentation models](https://pytorch.org/blog/pytorch-1.12-new-library-releases/#beta-object-detection-and-instance-segmentation), it was a hacky approach which made those transforms impossible to import from the TorchVision binary.

## The new Transforms API

The Transforms V2 API supports videos, bounding boxes, labels and segmentation masks meaning that it offers native support for many Computer Vision tasks. The new solution is a drop-in replacement:

<p align="center">
  <img src="" width="90%">
</p>

The new Transform Classes can receive any arbitrary number of inputs without enforcing specific order or structure:

<p align="center">
  <img src="" width="90%">
</p>

The Transform Classes make sure that they apply the same random transforms to all the inputs to ensure consistent results:

<p align="center">
  <img src="" width="90%">
</p>

<p align="center">
<b>Original</b>
</p>

<p align="center">
  <img src="" width="90%">
</p>

<p align="center">
<b>Rotated and Cropped</b>
</p>

The functional API has been updated to support all necessary signal processing kernels (resizing, cropping, affine transforms, padding etc) for all inputs:

<p align="center">
  <img src="" width="90%">
</p>

The API uses Tensor subclassing to wrap input, attach useful meta-data and dispatch to the right kernel. Once the Datasets V2 work is complete, which makes use of TorchData’s Data Pipes, the manual wrapping of input won’t be necessary. For now, users can manually wrap the input by:

<p align="center">
  <img src="" width="90%">
</p>

In addition to the new API, we now provide importable implementations for several data augmentations that are used in SoTA research such as [MixUp](https://github.com/pytorch/vision/blob/main/torchvision/prototype/transforms/_augment.py#L129), [CutMix](https://github.com/pytorch/vision/blob/main/torchvision/prototype/transforms/_augment.py#L152), [Large Scale Jitter](https://github.com/pytorch/vision/blob/main/torchvision/prototype/transforms/_geometry.py#L705), [SimpleCopyPaste](https://github.com/pytorch/vision/blob/main/torchvision/prototype/transforms/_augment.py#L197), [AutoAugmentation](https://github.com/pytorch/vision/blob/main/torchvision/prototype/transforms/_auto_augment.py) methods and [several](https://github.com/pytorch/vision/blob/main/torchvision/prototype/transforms/__init__.py) new Geometric, Colour and Type Conversion transforms.

The API continues to support both PIL and Tensor backends for Images, single or batched input and maintains JIT-scriptability on the functional API. It allows deferring the casting of images from `uint8` to `float` which can lead to performance benefits. It is currently available in the [prototype area](https://github.com/pytorch/vision/tree/main/torchvision/prototype/transforms) of TorchVision and can be imported from the nightly builds. The new API has been [verified](https://github.com/pytorch/vision/pull/6433#issuecomment-1256741233) to achieve the same accuracy as the previous implementation.