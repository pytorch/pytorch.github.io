---
layout: blog_detail
title: "Extending TorchVision’s Transforms to Object Detection, Segmentation & Video tasks"
author: Philip Meier, Victor Fomin, Vasilis Vryniotis
featured-img: "assets/images/Transforms-v2-feature-image.png"
---

TorchVision is extending its Transforms API! Here is what’s new:

- You can use them not only for Image Classification but also for Object Detection, Instance & Semantic Segmentation and Video Classification.
- You can import directly from TorchVision several SoTA data-augmentations such as MixUp, CutMix, Large Scale Jitter and SimpleCopyPaste.  
- You can use new functional transforms for transforming Videos, Bounding Boxes and Segmentation Masks.

The interface remains the same to assist the migration and adoption. The new API is currently in Prototype and we would love to get early feedback from you to improve its functionality. Please [reach out to us](https://github.com/pytorch/vision/issues/6753) if you have any questions or suggestions.

## Limitations of current Transforms

The stable Transforms API of TorchVision (aka V1) only supports single images. As a result it can only be used for classification tasks:

```Python
from torchvision import transforms

trans = transforms.Compose([
   transforms.ColorJitter(contrast=0.5),
   transforms.RandomRotation(30),
   transforms.CenterCrop(480),
])

imgs = trans(imgs)
```

The above approach doesn’t support Object Detection, Segmentation or Classification transforms that require the use of Labels (such as MixUp & CutMix). This limitation made any non-classification Computer Vision tasks second-class citizens as one couldn’t use the Transforms API to perform the necessary augmentations. Historically this made it difficult to train high-accuracy models using TorchVision’s primitives and thus our Model Zoo lagged by several points from SoTA.

To circumvent this limitation, TorchVision offered [custom implementations](https://github.com/pytorch/vision/blob/main/references/detection/transforms.py) in its reference scripts that show-cased how one could perform augmentations in each task. Though this practice enabled us to train high accuracy [classification](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/), [object detection & segmentation models](https://pytorch.org/blog/pytorch-1.12-new-library-releases/#beta-object-detection-and-instance-segmentation), it was a hacky approach which made those transforms impossible to import from the TorchVision binary.

## The new Transforms API

The Transforms V2 API supports videos, bounding boxes, labels and segmentation masks meaning that it offers native support for many Computer Vision tasks. The new solution is a drop-in replacement:

```Python
from torchvision.prototype import transforms


# Exactly the same interface as V1:
trans = transforms.Compose([
    transforms.ColorJitter(contrast=0.5),
    transforms.RandomRotation(30),
    transforms.CenterCrop(480),
])

imgs, bboxes, labels = trans(imgs, bboxes, labels)
```

The new Transform Classes can receive any arbitrary number of inputs without enforcing specific order or structure:

```Python
# Already supported:
trans(imgs)  # Image Classification
trans(videos)  # Video Tasks
trans(imgs_or_videos, labels)  # MixUp/CutMix-style Transforms
trans(imgs, bboxes, labels)  # Object Detection
trans(imgs, bboxes, masks, labels)  # Instance Segmentation
trans(imgs, masks)  # Semantic Segmentation
trans({"image": imgs, "box": bboxes, "tag": labels})  # Arbitrary Structure

# Future support:
trans(imgs, bboxes, labels, keypoints)  # Keypoint Detection
trans(stereo_images, disparities, masks)  # Depth Perception
trans(image1, image2, optical_flows, masks)  # Optical Flow
```

The Transform Classes make sure that they apply the same random transforms to all the inputs to ensure consistent results.

The functional API has been updated to support all necessary signal processing kernels (resizing, cropping, affine transforms, padding etc) for all inputs:


```Python
from torchvision.prototype.transforms import functional as F


# High-level dispatcher, accepts any supported input type, fully BC
F.resize(inpt, resize=[224, 224])
# Image tensor kernel
F.resize_image_tensor(img_tensor, resize=[224, 224], antialias=True) 
# PIL image kernel
F.resize_image_pil(img_pil, resize=[224, 224], interpolation=BILINEAR)
# Video kernel
F.resize_video(video, resize=[224, 224], antialias=True) 
# Mask kernel
F.resize_mask(mask, resize=[224, 224])
# Bounding box kernel
F.resize_bounding_box(bbox, resize=[224, 224], spatial_size=[256, 256])
```

The API uses Tensor subclassing to wrap input, attach useful meta-data and dispatch to the right kernel. Once the Datasets V2 work is complete, which makes use of TorchData’s Data Pipes, the manual wrapping of input won’t be necessary. For now, users can manually wrap the input by:

```Python
from torchvision.prototype import features

imgs = features.Image(images, color_space=ColorSpace.RGB)
vids = features.Video(videos, color_space=ColorSpace.RGB)
masks = features.Mask(target["masks"])
bboxes = features.BoundingBox(target["boxes"], format=BoundingBoxFormat.XYXY, spatial_size=imgs.spatial_size)
labels = features.Label(target["labels"], categories=["dog", "cat"])
```

In addition to the new API, we now provide importable implementations for several data augmentations that are used in SoTA research such as [MixUp](https://github.com/pytorch/vision/blob/main/torchvision/prototype/transforms/_augment.py#L129), [CutMix](https://github.com/pytorch/vision/blob/main/torchvision/prototype/transforms/_augment.py#L152), [Large Scale Jitter](https://github.com/pytorch/vision/blob/main/torchvision/prototype/transforms/_geometry.py#L705), [SimpleCopyPaste](https://github.com/pytorch/vision/blob/main/torchvision/prototype/transforms/_augment.py#L197), [AutoAugmentation](https://github.com/pytorch/vision/blob/main/torchvision/prototype/transforms/_auto_augment.py) methods and [several](https://github.com/pytorch/vision/blob/main/torchvision/prototype/transforms/__init__.py) new Geometric, Colour and Type Conversion transforms.

The API continues to support both PIL and Tensor backends for Images, single or batched input and maintains JIT-scriptability on the functional API. It allows deferring the casting of images from `uint8` to `float` which can lead to performance benefits. It is currently available in the [prototype area](https://github.com/pytorch/vision/tree/main/torchvision/prototype/transforms) of TorchVision and can be imported from the nightly builds. The new API has been [verified](https://github.com/pytorch/vision/pull/6433#issuecomment-1256741233) to achieve the same accuracy as the previous implementation.

## Current Limitations

Though the functional API (kernels) remain JIT-scriptable and fully-BC, the Transform Classes, though they offer the same interface, [can’t be scripted](https://github.com/pytorch/vision/issues/6711). This is because they use Tensor Subclassing and receive arbitrary number of inputs which are not supported by JIT. We are currently working to reduce the dispatching overhead of the new API and to improve the speed of existing kernels.

## An end-to-end example

Here is an example of the new API using the following [image](https://user-images.githubusercontent.com/5347466/195350223-8683ef25-1367-4292-9174-c15f85c7358e.jpg). It works both with PIL images and Tensors:

```Python
import PIL
from torchvision import io, utils
from torchvision.prototype import features, transforms as T
from torchvision.prototype.transforms import functional as F


# Defining and wrapping input to appropriate Tensor Subclasses
path = "COCO_val2014_000000418825.jpg"
img = features.Image(io.read_image(path), color_space=features.ColorSpace.RGB)
# img = PIL.Image.open(path)
bboxes = features.BoundingBox(
    [[2, 0, 206, 253], [396, 92, 479, 241], [328, 253, 417, 332],
     [148, 68, 256, 182], [93, 158, 170, 260], [432, 0, 438, 26],
     [422, 0, 480, 25], [419, 39, 424, 52], [448, 37, 456, 62],
     [435, 43, 437, 50], [461, 36, 469, 63], [461, 75, 469, 94],
     [469, 36, 480, 64], [440, 37, 446, 56], [398, 233, 480, 304],
     [452, 39, 463, 63], [424, 38, 429, 50]],
    format=features.BoundingBoxFormat.XYXY,
    spatial_size=F.get_spatial_size(img),
)
labels = features.Label([59, 58, 50, 64, 76, 74, 74, 74, 74, 74, 74, 74, 74, 74, 50, 74, 74])


# Defining and applying Transforms V2
trans = T.Compose(
    [
        T.ColorJitter(contrast=0.5),
        T.RandomRotation(30),
        T.CenterCrop(480),
    ]
)
img, bboxes, labels = trans(img, bboxes, labels)


# Visualizing results
viz = utils.draw_bounding_boxes(F.to_image_tensor(img), boxes=bboxes)
F.to_pil_image(viz).show()
```

## Development milestones and future work

Here is where we are in development:

- [x] ~~Design API~~
- [x] ~~Write Kernels for transforming Videos, Bounding Boxes, Masks and Labels~~
- [x] ~~Rewrite all existing Transform Classes (stable + references) on the new API:~~
  - [x] ~~Image Classification~~
  - [x] ~~Video Classification~~
  - [x] ~~Object Detection~~
  - [x] ~~Instance Segmentation~~
  - [x] ~~Semantic Segmentation~~
- [x] ~~Verify the accuracy of the new API for all supported Tasks and Backends~~
- [ ] Speed Benchmarks and Performance Optimizations (in progress - planned for Dec)
- [ ] Graduate from Prototype (planned for Q1)
- [ ] Add support of Depth Perception, Keypoint Detection, Optical Flow and more (future)

We are currently in the process of Benchmarking each Transform Class and Functional Kernel in order to measure and improve their performance. The scope includes optimizing existing kernels which will be adopted from V1. Early findings indicate that some improvements might need to be [upstreamed](https://github.com/pytorch/pytorch/pull/86361) on the C++ kernels of PyTorch Core. Our plan is to continue iterating throughout Q4 to improve the speed performance of the new API and enhance it with additional [SoTA transforms](https://github.com/pytorch/vision/issues/6323) with the help of the community.

We would love to get early [feedback](https://github.com/pytorch/vision/issues/6753) from you to improve its functionality. Please reach out to us if you have any questions or suggestions.