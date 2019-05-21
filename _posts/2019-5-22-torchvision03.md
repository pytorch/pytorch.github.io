# torchvision 0.3 released, adds new models, datasets and more..

PyTorch domain libraries like torchvision enable researchers to be more efficient by providing convenient access to common datasets and models that can be used to quickly create a SOTA baseline. Moreover, they also provide common abstractions to reduce boilerplate code that users might have to otherwise repeatedly write. The torchvision 0.3 release brings several new features including models for semantic segmentation, object detection, instance segmentation, and person keypoint detection, as well as custom C++ / CUDA ops specific to computer vision. 

<div class="text-center">
  <img src="{{ site.url }}/assets/images/tv_tutorial/tv_image03.png" width="100%">
</div>

### New features include:

*Reference training / evaluation scripts: *torchvision now provides, under the references/ folder, scripts for training and evaluation of the following tasks: classification, semantic segmentation, object detection, instance segmentation and person keypoint detection. These serve as a log of how to train a specific model and provide baseline training and evaluation scripts to quickly bootstrap research.

*torchvision ops: *torchvision now contains custom C++ / CUDA operators. Those operators are specific to computer vision, and make it easier to build object detection models. These operators currently do not support PyTorch script mode, but support for it is planned for in the next release. Some of the ops supported include: 

* roi_pool (and the module version RoIPool)
* roi_align (and the module version RoIAlign)
* nms, for non-maximum suppression of bounding boxes
* box_iou, for computing the intersection over union metric between two sets of bounding boxes
* box_area, for computing the area of a set of bounding boxes

**New models and datasets:** torchvision now adds support for object detection, instance segmentation and person keypoint detection models. In addition, several popular datasets have been added. Note: The API is currently experimental and might change in future versions of torchvision. New models include: 

#### Classification Models

* GoogLeNet (Inception v1) 
* MobileNet V2 
* ShuffleNet v2 
* ResNeXt-50 32x4d and ResNeXt-101 32x8d 

#### Segmentation Models

* Fully-Convolutional Network (FCN) with ResNet 101 backbone
* DeepLabV3 with ResNet 101 backbone

#### Detection Models

* Faster R-CNN R-50 FPN trained on COCO train2017 
* Mask R-CNN R-50 FPN trained on COCO train2017 
* Keypoint R-CNN R-50 FPN trained on COCO train2017 

#### Datasets

* Caltech101, Caltech256, and CelebA 
* ImageNet dataset 
* Semantic Boundaries Dataset
* VisionDataset as a base class for all datasets


In addition, we've added more image transforms, general improvements and bug fixes, as well as improved documentation. See the full release notes [here](https://github.com/pytorch/vision/releases) as well as this getting started tutorial [here](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html), which describes how to fine tune your own instance segmentation model on a custom dataset. 

Cheers!

Team PyTorch
