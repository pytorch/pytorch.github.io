---
layout: blog_detail
title: 'torchvision 0.3: segmentation, detection models, new datasets and more..'
author: Francisco Massa
redirect_from: /2019/05/23/torchvision03.html
---

PyTorch domain libraries like torchvision provide convenient access to common datasets and models that can be used to quickly create a state-of-the-art baseline. Moreover, they also provide common abstractions to reduce boilerplate code that users might have to otherwise repeatedly write. The torchvision 0.3 release brings several new features including models for semantic segmentation, object detection, instance segmentation, and person keypoint detection, as well as custom C++ / CUDA ops specific to computer vision.

<div class="text-center">
  <img src="{{ site.url }}/assets/images/torchvision_0.3_headline.png" width="100%">
</div>


### New features include:

**Reference training / evaluation scripts:** torchvision now provides, under the references/ folder, scripts for training and evaluation of the following tasks: classification, semantic segmentation, object detection, instance segmentation and person keypoint detection. These serve as a log of how to train a specific model and provide baseline training and evaluation scripts to quickly bootstrap research.

**torchvision ops:** torchvision now contains custom C++ / CUDA operators. Those operators are specific to computer vision, and make it easier to build object detection models. These operators currently do not support PyTorch script mode, but support for it is planned for in the next release. Some of the ops supported include:

* roi_pool (and the module version RoIPool)
* roi_align (and the module version RoIAlign)
* nms, for non-maximum suppression of bounding boxes
* box_iou, for computing the intersection over union metric between two sets of bounding boxes
* box_area, for computing the area of a set of bounding boxes

Here are a few examples on using torchvision ops:

```python
import torch
import torchvision

# create 10 random boxes
boxes = torch.rand(10, 4) * 100
# they need to be in [x0, y0, x1, y1] format
boxes[:, 2:] += boxes[:, :2]
# create a random image
image = torch.rand(1, 3, 200, 200)
# extract regions in `image` defined in `boxes`, rescaling
# them to have a size of 3x3
pooled_regions = torchvision.ops.roi_align(image, [boxes], output_size=(3, 3))
# check the size
print(pooled_regions.shape)
# torch.Size([10, 3, 3, 3])

# or compute the intersection over union between
# all pairs of boxes
print(torchvision.ops.box_iou(boxes, boxes).shape)
# torch.Size([10, 10])
```


**New models and datasets:** torchvision now adds support for object detection, instance segmentation and person keypoint detection models. In addition, several popular datasets have been added. Note: The API is currently experimental and might change in future versions of torchvision. New models include:

### Segmentation Models

The 0.3 release also contains models for dense pixelwise prediction on images.
It adds FCN and DeepLabV3 segmentation models, using a ResNet50 and ResNet101 backbones.
Pre-trained weights for ResNet101 backbone are available, and have been trained on a subset of COCO train2017, which contains the same 20 categories as those from Pascal VOC.

The pre-trained models give the following results on the subset of COCO val2017 which contain the same 20 categories as those present in Pascal VOC:

Network | mean IoU | global pixelwise acc
-- | -- | --
FCN ResNet101 | 63.7 | 91.9
DeepLabV3 ResNet101 | 67.4 | 92.4

### Detection Models

Network | box AP | mask AP | keypoint AP
-- | -- | -- | --
Faster R-CNN ResNet-50 FPN trained on COCO | 37.0 |   |  
Mask R-CNN ResNet-50 FPN trained on COCO | 37.9 | 34.6 |  
Keypoint R-CNN ResNet-50 FPN trained on COCO | 54.6 |   | 65.0

The implementations of the models for object detection, instance segmentation and keypoint detection are fast, specially during training.

In the following table, we use 8 V100 GPUs, with CUDA 10.0 and CUDNN 7.4 to report the results. During training, we use a batch size of 2 per GPU, and during testing a batch size of 1 is used.

For test time, we report the time for the model evaluation and post-processing (including mask pasting in image), but not the time for computing the precision-recall.

Network | train time (s / it) | test time (s / it) | memory (GB)
-- | -- | -- | --
Faster R-CNN ResNet-50 FPN | 0.2288 | 0.0590 | 5.2
Mask R-CNN ResNet-50 FPN | 0.2728 | 0.0903 | 5.4
Keypoint R-CNN ResNet-50 FPN | 0.3789 | 0.1242 | 6.8


You can load and use pre-trained detection and segmentation models with a few lines of code

```python
import torchvision

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# set it to evaluation mode, as the model behaves differently
# during training and during evaluation
model.eval()

image = PIL.Image.open('/path/to/an/image.jpg')
image_tensor = torchvision.transforms.functional.to_tensor(image)

# pass a list of (potentially different sized) tensors
# to the model, in 0-1 range. The model will take care of
# batching them together and normalizing
output = model([image_tensor])
# output is a list of dict, containing the postprocessed predictions
```

### Classification Models

The following classification models were added:

* GoogLeNet (Inception v1)
* MobileNet V2
* ShuffleNet v2
* ResNeXt-50 32x4d and ResNeXt-101 32x8d

### Datasets

The following datasets were added:

* Caltech101, Caltech256, and CelebA
* ImageNet dataset (improving on ImageFolder, provides class-strings)
* Semantic Boundaries Dataset
* VisionDataset as a base class for all datasets


In addition, we've added more image transforms, general improvements and bug fixes, as well as improved documentation.

**See the full release notes [here](https://github.com/pytorch/vision/releases) as well as this getting started tutorial [on Google Colab here](https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb), which describes how to fine tune your own instance segmentation model on a custom dataset.**

Cheers!

Team PyTorch
