---
layout: blog_detail
title: 'Everything You Need To Know About Torchvision’s SSD Implementation'
author: Vasilis Vryniotis
---

In TorchVision v0.10, we’ve released two new Object Detection models based on the SSD architecture. Our plan is to cover the key implementation details of the algorithms along with information on how they were trained in a two-part article.

In part 1 of the series, we will focus on the original implementation of the SSD algorithm as described on the [Single Shot MultiBox Detector paper](https://arxiv.org/abs/1512.02325). We will briefly give a high-level description of how the algorithm works, then go through its main components, highlight key parts of its code, and finally discuss how we trained the released model. Our goal is to cover all the necessary details to reproduce the model including those optimizations which are not covered on the paper but are part on the [original implementation](https://github.com/weiliu89/caffe/tree/ssd).

# How Does SSD Work?

Reading the aforementioned paper is highly recommended but here is a quick oversimplified refresher. Our target is to detect the locations of objects in an image along with their categories. Here is the Figure 5 from the [SSD paper](https://arxiv.org/abs/1512.02325) with prediction examples of the model:

<div class="text-center">
  <img src="{{ site.url }}/assets/images/prediction examples.png" width="100%">
</div>

The SSD algorithm uses a CNN backbone, passes the input image through it and takes the convolutional outputs from different levels of the network. The list of these outputs are called feature maps. These feature maps are then passed through the Classification and Regression heads which are responsible for predicting the class and the location of the boxes. 

Since the feature maps of each image contain outputs from different levels of the network, their size varies and thus they can capture objects of different dimensions. On top of each, we tile several default boxes which can be thought as our rough prior guesses. For each default box, we predict whether there is an object (along with its class) and its offset (correction over the original location). During training time, we need to first match the ground truth to the default boxes and then we use those matches to estimate our loss. During inference, similar prediction boxes are combined to estimate the final predictions. 

# The SSD Network Architecture

In this section, we will discuss the key components of SSD. Our code follows closely [the paper](https://arxiv.org/abs/1512.02325) and makes use of many of the undocumented optimizations included in [the official implementation](https://github.com/weiliu89/caffe/tree/ssd).

### DefaultBoxGenerator

The [DefaultBoxGenerator class](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/anchor_utils.py#L134) is responsible for generating the default boxes of SSD and operates similarly to the [AnchorGenerator](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/anchor_utils.py#L9) of FasterRCNN (for more info on their differences see pages 4-6 of the paper). It produces a set of predefined boxes of specific width and height which are tiled across the image and serve as the first rough prior guesses of where objects might be located. Here is Figure 1 from the SSD paper with a visualization of ground truths and default boxes:

<div class="text-center">
  <img src="{{ site.url }}/assets/images/visualization of ground truths.png" width="100%">
</div>

The class is parameterized by a set of hyperparameters that control [their shape](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/anchor_utils.py#L139) and [tiling](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/anchor_utils.py#L140-L149). The implementation will provide [automatically good guesses](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/anchor_utils.py#L162-L171) with the default parameters for those who want to experiment with new backbones/datasets but one can also pass [optimized custom values](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/anchor_utils.py#L144-L147).

### SSDMatcher

The [SSDMatcher class](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/_utils.py#L348) extends the standard [Matcher](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/_utils.py#L227) used by FasterRCNN and it is responsible for matching the default boxes to the ground truth. After estimating the [IoUs of all combinations](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L349), we use the matcher to find for each default box the best [candidate](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/_utils.py#L296) ground truth with overlap higher than the [IoU threshold](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/_utils.py#L350-L351). The SSD version of the matcher has an extra step to ensure that each ground truth is matched with the default box that has the [highest overlap](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/_utils.py#L356-L360). The results of the matcher are used in the loss estimation during the training process of the model.

### Classification and Regression Heads

The [SSDHead class](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L38) is responsible for initializing the Classification and Regression parts of the network. Here are a few notable details about their code:

* Both the [Classification](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L90) and the [Regression](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L99) head inherit from the [same class](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L51) which is responsible for making the predictions for each feature map. 
* Each level of the feature map uses a separate 3x3 Convolution to estimate the [class logits](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L92-L94) and [box locations](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L101-L103). 
* The [number of predictions](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L79) that each head makes per level depends on the number of default boxes and the sizes of the feature maps.

### Backbone Feature Extractor

The [feature extractor](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L413) reconfigures and enhances a standard VGG backbone with extra layers as depicted on the Figure 2 of the SSD paper: 

<div class="text-center">
  <img src="{{ site.url }}/assets/images/feature extractor.png" width="100%">
</div>

The class supports all VGG models of TorchVision and one can create a similar extractor class for other types of CNNs (see [this example for ResNet](https://github.com/pytorch/vision/blob/644bdcdc438c1723714950d0771da76333b53954/torchvision/models/detection/ssd.py#L600)). Here are a few implementation details of the class:

* [Patching](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L419-L420) the ```ceil_mode parameter``` of the 3rd Maxpool layer is necessary to get the same feature map sizes as the paper. This is due to small differences between PyTorch and the original Caffe implementation of the model. 
* It adds a series of [extra feature layers](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L430-L456)on top of VGG. If the highres parameter is ```True``` during its construction, it will append an [extra convolution](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L457-L464). This is useful for the SSD512 version of the model.
* As discussed on section 3 of the paper, the fully connected layers of the original VGG are converted to convolutions with the [first one using Atrous](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L469). Moreover maxpool5’s stride and kernel size is [modified](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L468).
* As described on section 3.1, L2 normalization is used on the [output of conv4_3](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L484) and a set of [learnable weights](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L422-L423) are introduced to control its scaling.

### SSD Algorithm

The final key piece of the implementation is on the [SSD class](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L108). Here are some notable details:

* The algorithm is [parameterized](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L167-L176) by a set of arguments similar to other detection models. The mandatory parameters are: the backbone which is responsible for [estimating the feature maps](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L137-L139), the ```anchor_generator``` which should be a [configured instance](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L140-L141) of the ```DefaultBoxGenerator``` class, the size to which the [input images](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L142-L143) will be resized and the ```num_classes``` for classification [excluding](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L144) the background.
* If a [head](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L150-L151) is not provided, the constructor will [initialize](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L194) the default ```SSDHead```. To do so, we need to know the number of output channels for each feature map produced by the backbone. Initially we try to [retrieve this information](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L186) from the backbone but if not available we will [dynamically estimate it](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L189).
* The algorithm [reuses](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L183) the standard [BoxCoder class](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/_utils.py#L129) used by other Detection models. The class is responsible for [encoding and decoding](https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/) the bounding boxes and is configured to use the same prior variances as the [original implementation](https://github.com/weiliu89/caffe/blob/2c4e4c2899ad7c3a997afef2c1fbac76adca1bad/examples/ssd/ssd_coco.py#L326).
* Though we reuse the standard [GeneralizedRCNNTransform class](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/transform.py#L64) to resize and normalize the input images, the SSD algorithm [configures](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L203-L204) it to ensure that the image size will remain fixed. 

Here are the two core methods of the implementation:

* The ```compute_loss``` method estimates the standard Multi-box loss as described on page 5 of the SSD paper. It uses the [smooth L1 loss](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L244) for regression and the standard [cross-entropy loss](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L262-L266) with [hard-negative sampling](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L268-L276) for classification. 
* As in all detection models, the forward method currently has different behaviour depending on whether the model is on training or eval mode. It starts by [resizing & normalizing the input images](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L309-L310) and then [passes them through the backbone](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L324-L325) to get the feature maps. The feature maps are then [passed through the head](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L331-L332) to get the predictions and then the method [generates the default boxes](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L334-L335). 
    * If the model is on [training mode](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L339-L352), the forward will estimate the [IoUs of the default boxes with the ground truth](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L349), use the ```SSDmatcher``` to [produce matches](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L350) and finally [estimate the losses](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L352) by calling the ```compute_loss method```.
    * If the model is on [eval mode](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L353-L355), we first select the best detections by keeping only the ones that [pass the score threshold](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L384), select the [most promising boxes](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L388-L391) and run NMS to [clean up and select](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L401-L403) the best predictions. Finally we [postprocess the predictions](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L355) to resize them to the original image size.

# The SSD300 VGG16 Model

The SSD is a family of models because it can be configured with different backbones and different Head configurations. In this section, we will focus on the provided [SSD pre-trained model](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L522-L523). We will discuss the details of its configuration and the training process used to reproduce the reported results.

### Training process

The model was trained using the COCO dataset and all of its hyper-parameters and scripts can be found in our [references](https://github.com/pytorch/vision/blob/e35793a1a4000db1f9f99673437c514e24e65451/references/detection/README.md#ssd300-vgg16) folder. Below we provide details on the most notable aspects of the training process.

### Paper Hyperparameters

In order to achieve the best possible results on COCO, we adopted the hyperparameters described on the section 3 of the paper concerning the optimizer configuration, the weight regularization etc. Moreover we found it useful to adopt the optimizations that appear in the [official implementation](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_coco.py#L310-L321) concerning the [tiling configuration](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L579-L581) of the DefaultBox generator. This optimization was not described in the paper but it was crucial for improving the detection precision of smaller objects. 

### Data Augmentation

Implementing the [SSD Data Augmentation strategy](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/references/detection/transforms.py#L20-L239) as described on page 6 and page 12 of the paper was critical to reproducing the results. More specifically the use of random “Zoom In” and “Zoom Out” transformations make the model robust to various input sizes and improve its precision on the small and medium objects. Finally since the VGG16 has quite a few parameters, the photometric distortions [included in the augmentations](https://github.com/pytorch/vision/blob/43d772067fe77965ec8fc49c799de5cea44b8aa2/references/detection/presets.py#L11-L18) have a regularization effect and help avoid the overfitting. 

### Weight Initialization & Input Scaling

Another aspect that we found beneficial was to follow the [weight initialization scheme](https://github.com/intel/caffe/blob/master/models/intel_optimized_models/ssd/VGGNet/coco/SSD_300x300/train.prototxt) proposed by the paper. To do that, we had to adapt our input scaling method by [undoing the 0-1 scaling](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L583-L587) performed by ```ToTensor()``` and use [pre-trained ImageNet weights](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L24-L26) fitted with this scaling (shoutout to [Max deGroot](https://github.com/amdegroot) for providing them in his repo). All the weights of new convolutions were [initialized using Xavier](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L30-L35) and their biases were set to zero. After initialization, the network was [trained end-to-end](https://github.com/pytorch/vision/blob/33db2b3ebfdd2f73a9228f430fa7dd91c3b18078/torchvision/models/detection/ssd.py#L571-L572). 

### LR Scheme

As reported on the paper, after applying aggressive data augmentations it’s necessary to train the models for longer. Our experiments confirm this and we had to tweak the Learning rate, batch sizes and overall steps to achieve the best results. Our [proposed learning scheme](https://github.com/pytorch/vision/blob/e35793a1a4000db1f9f99673437c514e24e65451/references/detection/README.md#ssd300-vgg16) is configured to be rather on the safe side, showed signs of plateauing between the steps and thus one is likely to be able to train a similar model by doing only 66% of our epochs.

# Breakdown of Key Accuracy Improvements

It is important to note that implementing a model directly from a paper is an iterative process that circles between coding, training, bug fixing and adapting the configuration until we match the accuracies reported on the paper. Quite often it also involves simplifying the training recipe or enhancing it with more recent methodologies. It is definitely not a linear process where incremental accuracy improvements are achieved by improving a single direction at a time but instead involves exploring different hypothesis, making incremental improvements in different aspects and doing a lot of backtracking. 

With that in mind, below we try to summarize the optimizations that affected our accuracy the most. We did this by grouping together the various experiments in 4 main groups and attributing the experiment improvements to the closest match. Note that the Y-axis of the graph starts from 18 instead from 0 to make the difference between optimizations more visible:

<div class="text-center">
  <img src="{{ site.url }}/assets/images/Key optimizations for improving the mAP of SSD300 VGG16.png" width="100%">
</div>

| Model Configuration | mAP delta | mAP | 
| ------------- | ------------- |  ------------- |
| Baseline with "FasterRCNN-style" Hyperparams | - | 19.5 | 
| + Paper Hyperparams | 1.6 | 21.1 | 
| + Data Augmentation | 1.8 | 22.9 | 
| + Weight Initialization & Input Scaling | 1 | 23.9 | 
| + LR scheme | 1.2 | 25.1 | 

Our final model achieves an mAP of 25.1 and reproduces exactly the COCO results reported on the paper. Here is a [detailed breakdown](https://github.com/pytorch/vision/pull/3403) of the accuracy metrics.


We hope you found the part 1 of the series interesting. On the part 2, we will focus on the implementation of SSDlite and discuss its differences from SSD. Until then, we are looking forward to your feedback.
