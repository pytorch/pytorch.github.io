---
layout: blog_detail
title: "Extending TorchVision’s Transforms to Object Detection, Segmentation & Video tasks"
author: Philip Meier, Victor Fomin, Vasilis Vryniotis, Nicolas Hug
featured-img: "assets/images/Transforms-v2-feature-image.png"
---

**Note**: A previous version of this post was published in November 2022. We have updated this post with the most up-to-date info, in view of the upcoming 0.15 release of torchvision in March 2023, jointly with PyTorch 2.0.

TorchVision is extending its Transforms API! Here is what’s new:

- You can use them not only for Image Classification but also for Object Detection, Instance & Semantic Segmentation and Video Classification.
- You can use new functional transforms for transforming Videos, Bounding Boxes and Segmentation Masks.


The API is completely backward compatible with the previous one, and remains the same to assist the migration and adoption. We are now releasing this new API as Beta in the torchvision.transforms.v2 namespace, and we would love to get early feedback from you to improve its functionality. Please [_reach out to us_](https://github.com/pytorch/vision/issues/6753) if you have any questions or suggestions.

## Limitations of current Transforms

The existing Transforms API of TorchVision (aka V1) only supports single images. As a result it can only be used for classification tasks:

```python
from torchvision import transforms
trans = transforms.Compose([
   transforms.ColorJitter(contrast=0.5),
   transforms.RandomRotation(30),
   transforms.CenterCrop(480),
])
imgs = trans(imgs)
```

The above approach doesn’t support Object Detection nor Segmentation. This limitation made any non-classification Computer Vision tasks second-class citizens as one couldn’t use the Transforms API to perform the necessary augmentations. Historically this made it difficult to train high-accuracy models using TorchVision’s primitives and thus our Model Zoo lagged by several points from SoTA.

To circumvent this limitation, TorchVision offered [_custom implementations_](https://github.com/pytorch/vision/blob/main/references/detection/transforms.py) in its reference scripts that show-cased how one could perform augmentations in each task. Though this practice enabled us to train high accuracy [_classification_](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/), [_object detection & segmentation_](https://pytorch.org/blog/pytorch-1.12-new-library-releases/#beta-object-detection-and-instance-segmentation) models, it was a hacky approach which made those transforms impossible to import from the TorchVision binary.

## The new Transforms API

The Transforms V2 API supports videos, bounding boxes, and segmentation masks meaning that it offers native support for many Computer Vision tasks. The new solution is a drop-in replacement:

```python
import torchvision.transforms.v2 as transforms

# Exactly the same interface as V1:
trans = transforms.Compose([
    transforms.ColorJitter(contrast=0.5),
    transforms.RandomRotation(30),
    transforms.CenterCrop(480),
])
imgs, bboxes, labels = trans(imgs, bboxes, labels)
```

The new Transform Classes can receive any arbitrary number of inputs without enforcing specific order or structure:

```python
# Already supported:
trans(imgs)  # Image Classification
trans(videos)  # Video Tasks
trans(imgs, bboxes, labels)  # Object Detection
trans(imgs, bboxes, masks, labels)  # Instance Segmentation
trans(imgs, masks)  # Semantic Segmentation
trans({"image": imgs, "box": bboxes, "tag": labels})  # Arbitrary Structure

# Future support:
trans(imgs, bboxes, labels, keypoints)  # Keypoint Detection
trans(stereo_images, disparities, masks)  # Depth Perception
trans(image1, image2, optical_flows, masks)  # Optical Flow
trans(imgs_or_videos, labels)  # MixUp/CutMix-style Transforms
```

The Transform Classes make sure that they apply the same random transforms to all the inputs to ensure consistent results.

The functional API has been updated to support all necessary signal processing kernels (resizing, cropping, affine transforms, padding etc) for all inputs:

```python
from torchvision.transforms.v2 import functional as F


# High-level dispatcher, accepts any supported input type, fully BC
F.resize(inpt, size=[224, 224])
# Image tensor kernel
F.resize_image_tensor(img_tensor, size=[224, 224], antialias=True) 
# PIL image kernel
F.resize_image_pil(img_pil, size=[224, 224], interpolation=BILINEAR)
# Video kernel
F.resize_video(video, size=[224, 224], antialias=True) 
# Mask kernel
F.resize_mask(mask, size=[224, 224])
# Bounding box kernel
F.resize_bounding_box(bbox, size=[224, 224], spatial_size=[256, 256])
```

Under the hood, the API uses Tensor subclassing to wrap the input, attach useful meta-data and dispatch to the right kernel. For your data to be compatible with these new transforms, you can either use the provided dataset wrapper which should work with most of torchvision built-in datasets, or your can wrap your data manually into Datapoints:

```python
from torchvision.datasets import wrap_dataset_for_transforms_v2
ds = CocoDetection(..., transforms=v2_transforms)
ds = wrap_dataset_for_transforms_v2(ds) # data is now compatible with transforms v2!

# Or wrap your data manually using the lower-level Datapoint classes:
from torchvision import datapoints

imgs = datapoints.Image(images)
vids = datapoints.Video(videos)
masks = datapoints.Mask(target["masks“])
bboxes = datapoints.BoundingBox(target["boxes“], format=”XYXY”, spatial_size=imgs.shape)
```


In addition to the new API, we now provide importable implementations for several data augmentations that are used in SoTA research such as [_Large Scale Jitter_](https://github.com/pytorch/vision/blob/928b05cad36eadb13e169f03028767c8bcd1f21d/torchvision/transforms/v2/_geometry.py#L1109), [_AutoAugmentation_](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_auto_augment.py) methods and [_several_](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/__init__.py) new Geometric, Color and Type Conversion transforms.

The API continues to support both PIL and Tensor backends for Images, single or batched input and maintains JIT-scriptability on both the functional and class APIs.. The new API has been [_verified_](https://github.com/pytorch/vision/pull/6433#issuecomment-1256741233) to achieve the same accuracy as the previous implementation.

## An end-to-end example

 Here is an example of the new API using the following [_image_](https://user-images.githubusercontent.com/5347466/195350223-8683ef25-1367-4292-9174-c15f85c7358e.jpg). It works both with PIL images and Tensors. For more examples and tutorials, [_take a look at our gallery!_](https://pytorch.org/vision/0.15/auto_examples/index.html)


```python
from torchvision import io, utils
from torchvision import datapoints
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F

# Defining and wrapping input to appropriate Tensor Subclasses
path = "COCO_val2014_000000418825.jpg"
img = datapoints.Image(io.read_image(path))
# img = PIL.Image.open(path)
bboxes = datapoints.BoundingBox(
    [[2, 0, 206, 253], [396, 92, 479, 241], [328, 253, 417, 332],
     [148, 68, 256, 182], [93, 158, 170, 260], [432, 0, 438, 26],
     [422, 0, 480, 25], [419, 39, 424, 52], [448, 37, 456, 62],
     [435, 43, 437, 50], [461, 36, 469, 63], [461, 75, 469, 94],
     [469, 36, 480, 64], [440, 37, 446, 56], [398, 233, 480, 304],
     [452, 39, 463, 63], [424, 38, 429, 50]],
    format=datapoints.BoundingBoxFormat.XYXY,
    spatial_size=F.get_spatial_size(img),
)
labels = [59, 58, 50, 64, 76, 74, 74, 74, 74, 74, 74, 74, 74, 74, 50, 74, 74]
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

- [x] Design API
- [x] Write Kernels for transforming Videos, Bounding Boxes, Masks and Labels
- [x] Rewrite all existing Transform Classes (stable + references) on the new API:
  - [x] Image Classification
  - [x] Video Classification
  - [x] Object Detection
  - [x] Instance Segmentation
  - [x] Semantic Segmentation
- [x] Verify the accuracy of the new API for all supported Tasks and Backends
- [x] Speed Benchmarks and Performance Optimizations (in progress - planned for Dec)
- [x] Graduate from Prototype (planned for Q1)
- [ ] Add support of Depth Perception, Keypoint Detection, Optical Flow and more (future)
- [ ] Add smooth support for batch-wise transforms like MixUp and CutMix


We would love to get [_feedback_](https://github.com/pytorch/vision/issues/6753) from you to improve its functionality. Please reach out to us if you have any questions or suggestions.
