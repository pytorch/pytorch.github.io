---
layout: blog_detail
title: 'An overview of the ML models introduced in TorchVision v0.9'
author: Team PyTorch 
---

TorchVision v0.9 has been [released](https://github.com/pytorch/vision/releases) and it is packed with numerous new Machine Learning models and features, speed improvements and bug fixes. In this blog post, we provide a quick overview of the newly introduced ML models and discuss their key features and characteristics.

### Classification
* **MobileNetV3 Large & Small:** These two classification models are optimized for Mobile use-cases and are used as backbones on other Computer Vision tasks. The implementation of the new [MobileNetV3 architecture](https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv3.py) supports the Large & Small variants and the depth multiplier parameter as described in the [original paper](https://arxiv.org/pdf/1905.02244.pdf). We offer pre-trained weights on ImageNet for both Large and Small networks with depth multiplier 1.0 and resolution 224x224. Our previous [training recipes](https://github.com/pytorch/vision/tree/master/references/classification#mobilenetv3-large--small) have been updated and can be used to easily train the models from scratch (shoutout to Ross Wightman for inspiring some of our training configuration). The Large variant offers a [competitive accuracy](https://github.com/pytorch/vision/blob/master/docs/source/models.rst#classification) comparing to ResNet50 while being over 6x faster on CPU, meaning that it is a good candidate for applications where speed is important. For applications where speed is critical, one can sacrifice further accuracy for speed and use the Small variant which is 15x faster than ResNet50.

* **Quantized MobileNetV3 Large:** The quantized version of MobilNetV3 Large reduces the number of parameters by 45% and it is roughly 2.5x faster than the non-quantized version while remaining competitive in [terms of accuracy](https://github.com/pytorch/vision/blob/master/docs/source/models.rst#quantized-models). It was fitted on ImageNet using Quantization Aware Training by iterating on the non-quantized version and it can be trained from scratch using the existing [reference scripts](https://github.com/pytorch/vision/tree/master/references/classification#quantized).

**Usage:**
```
model = torchvision.models.mobilenet_v3_large(pretrained=True)
# model = torchvision.models.mobilenet_v3_small(pretrained=True)
# model = torchvision.models.quantization.mobilenet_v3_large(pretrained=True)
model.eval()
predictions = model(img)
```
### Object Detection
* **Faster R-CNN MobileNetV3-Large FPN:** Combining the MobileNetV3 Large backbone with a Faster R-CNN detector and a Feature Pyramid Network leads to a highly accurate and fast object detector. The pre-trained weights are fitted on COCO 2017 using the provided reference [scripts](https://github.com/pytorch/vision/tree/master/references/detection#faster-r-cnn-mobilenetv3-large-fpn) and the model is 5x faster on CPU than the equivalent ResNet50 detector while remaining competitive in [terms of accuracy](https://github.com/pytorch/vision/blob/master/docs/source/models.rst#object-detection-instance-segmentation-and-person-keypoint-detection). 
* **Faster R-CNN MobileNetV3-Large 320 FPN:** This is an iteration of the previous model that uses reduced resolution (min_size=320 pixel) and sacrifices accuracy for speed. It is 25x faster on CPU than the equivalent ResNet50 detector and thus it is good for real mobile use-cases.

**Usage:**
```
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
# model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
model.eval()
predictions = model(img)
```
### Semantic Segmentation
* **DeepLabV3 with Dilated MobileNetV3 Large Backbone:** A dilated version of the MobileNetV3 Large backbone combined with DeepLabV3 helps us build a highly accurate and fast semantic segmentation model. The pre-trained weights are fitted on COCO 2017 using our [standard training recipes](https://github.com/pytorch/vision/tree/master/references/segmentation#deeplabv3_mobilenet_v3_large). The final model has the [same accuracy](https://github.com/pytorch/vision/blob/master/docs/source/models.rst#semantic-segmentation) as the FCN ResNet50 but it is 8.5x faster on CPU and thus making it an excellent replacement for the majority of applications.
* **Lite R-ASPP with Dilated MobileNetV3 Large Backbone:** We introduce the implementation of a new segmentation head called Lite R-ASPP and combine it with the dilated MobileNetV3 Large backbone to build a very fast segmentation model. The new model sacrifices some accuracy to achieve a 15x speed improvement comparing to the previously most lightweight segmentation model which was the FCN ResNet50.

**Usage:**
```
model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
# model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained=True)
model.eval()
predictions = model(img)
```
In the near future we plan to publish an article that covers the details of how the above models were trained and discuss their tradeoffs and design choices. Until then we encourage you to try out the new models and provide your feedback.
