---
layout: blog_detail
title: "Accelerate PyTorch Models Using Quantization Techniques with Intel Extension for PyTorch"
author: Intel
---

## Overview

PyTorch is a Python-based framework for developing deep learning models. It is one of the most popular industry-standard AI frameworks and is used for a wide variety of computer vision and natural language processing applications. PyTorch was developed by Meta and is now part of The Linux Foundation. Intel works with the open source PyTorch project to optimize the PyTorch framework for Intel® hardware. The newest optimizations and features are first released in Intel® Extension for PyTorch before upstreaming them into PyTorch. The Intel extension provides quantization features to deliver good accuracy results for large deep learning models.

This article introduces quantization, types of quantization, and demonstrates a code sample on how to accelerate PyTorch-based models by applying Intel Extension for PyTorch quantization.


## What Is Quantization?

Quantization is a systematic reduction of the precision of all or several layers within the model. This means a higher-precision type (like single precision floating-point (FP32) that is mostly used in deep learning) is converted into a lower-precision type, such as FP16 (16 bits) or int8 (8 bits).

This helps to achieve:

* Lower memory bandwidth
* Lower storage
* Higher performance with minimum to zero accuracy loss

Quantization is especially important with large models such as those based on the Transformer architecture (like BERT or GPT).

There are two types of quantization:

* Static: This quantizes the weights and activations of the model, and is used when memory bandwidth and compute savings are important.
* Dynamic: The weights are quantized ahead of time, but the activations are dynamically quantized during inference.


## How to Perform Static Quantization and Dynamic Quantization

The Intel extension extends PyTorch with up-to-date features and optimizations for an extra performance boost on Intel hardware.

[Installation Instructions for Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch#installation)

The extension can be loaded as a Python module or linked as a C++ library. Python users can enable it dynamically by importing **intel_extension_for_pytorch**. The extension provides built-in quantization to deliver good statistical accuracy for most popular deep learning workloads including convolutional neural networks (CNN), natural language processing (NLP), and recommendation models. The quantization functionality in the Intel extension currently supports post-training quantization.

**To quantize the existing FP32 model to an int8 model using static quantization:**

1. Prepare the quantization configuration. For default static quantization configuration, use **ipex.quantization.default_static_qconfig**.
2. Prepare the model for calibration using the **ipex.quantization.prepare** method.
3. Perform calibration against the dataset. This calibration is specific for static quantization as it needs the representative dataset to determine the optimal quantization parameters, so the user should provide data to the model in batches to calibrate it.
4. Convert the model from FP32 to int8 using the **ipex.quantization.convert** method. This function converts the FP32 model to int8 based on the applied calibration and configuration.

**To quantize the existing FP32 model to an int8 model using dynamic quantization, which is similar to static quantization:**

1. Prepare the quantization configuration. For default dynamic quantization configuration, use **ipex.quantization.default_dynamic_qconfig**.
2. Prepare the FP32 model by using the **ipex.quantization.prepare** method. Provide the parameters, such as FP32 model to quantize, the prepared configuration, example inputs, and information.
3. Convert the model from FP32 to int8 using the **ipex.quantization.convert** method. The input model is the model prepared in Step 2.


## Code Sample


### Dataset

For static quantization, the model is calibrated with the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The CIFAR-10 is a subset of the 80 million [tiny images dataset](https://groups.csail.mit.edu/vision/TinyImages/) collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.

This dataset contains 60,000 images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and track). Every class has exactly 6,000 images. All images are 32 x 32 pixels and are colored. Also, the classes are completely mutually exclusive, which means there is no overlapping between classes.


### Implementation

The [code sample](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Features-and-Functionality/IntelPytorch_Quantization) demonstrates how to quantize (using static and dynamic quantization) a ResNet*-50 model using Intel Extension for PyTorch. The following steps are implemented in the code sample:


#### Download and Prepare the Dataset

Here, we use the CIFAR-10 dataset available in torchvision.

1. To make data fit the model:

* Transform the data.
* Change the size of the images from 32 x 32 pixels to 224 x 224 pixels.
* Convert them to tensors.
* Normalize them.

{:start="2"}
2. Prepare transformations of the dataset as shown: 

```
transform = torchvision.transforms.Compose([
torchvision.transforms.Resize((224, 224)),
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

```

{:start="3"}
3. Initialize the dataset.
 
```
test_dataset = torchvision.datasets.CIFAR10(root=DATA, train=False, transform=transform, download=Ture)
```


#### Prepare the Data Loader

To load a dataset for static quantization calibration in specific size batches, create the loader as shown:


```
calibration_data_loader = torch.utils.data.DataLoader(
dataset=test_dataset,
batch_size=128
)
```



#### Create the Model

Use the pretrained ResNet-50 model available in the Torchvision library with default weights. The prepared model is FP32.


```
model_fp32 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
```



#### Apply Static Quantization

Create a **staticQuantize** function that implements the steps described previously.



1. To perform static quantization, we need:

* FP32 model loaded earlier
* Example data
* Calibration dataset

{:start="2"}
2. Prepare the quantization configuration:

```
config_static = ipex.quantization.default_static_qconfig
```

In this code sample, we are using the default quantization configuration, but you can also define your own. \
 
{:start="3"}
3. Prepare the model using the declared configuration:


```
prepared_model_static = prepare(model_fp32,
qconfig_static,
example_inputs=data,
inplace=False)
```

{:start="4"}
4. Calibrate the model with the calibration dataset. Feed the model with successive batches of data from the dataset.


```
for batch_idx, (data, target) in enumerate(calibration_data_loader):
prepared_model_static(data)
if batch_idx % 10 == 0:
print("Batch %d/%d complete, continue ..." %(batch_idx+1, len(calibration_data_loader)))
```

{:start="5"}
5. Convert the model.

```
converted_model_static = convert(prepared_model_static)
```


#### Apply Dynamic Quantization

Create the **dynamicQuantize** function similar to the **staticQuantize** function.

1. To perform dynamic quantization, we only need:

* The FP32 model loaded earlier
* Example data

{:start="2"}
2. Prepare the quantization configuration:

```
qconfig_dynamic = ipex.quantization.default_dynamic_qconfig
```

{:start="3"}
3. Prepare the model.

```
prepared_model_dynamic = prepare(model_fp32,
qconfig_dynamic,
example_inputs=data,
inplace=False)
```

{:start="4"}
4. Convert the model from FP32 to int8.

```
converted_model_dynamic = convert(prepared_model_dynamic)
```

In this way, two functions are created to take advantage of the optimizations that quantization offers:

* **DynamicQuantize** for dynamic quantization of models
* **StaticQuantize** for static model quantization


## Next Steps

Get started with Intel Extension for PyTorch quantization today and use it to achieve better accuracy results for deep learning workloads. Additionally, [Intel® Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html?cid=sem&source=sa360&campid=2023_q2_iags_us_iagsoapie_iagsoapiee_awa_text-link_exact_cd_dpd-oneapi-intel_neural_compressor_3500107853_google_div_oos_non-pbm_intel&ad_group=ai_model_compression_exact&intel_term=neural+compressor&sa360id=43700076378213630&gclid=CjwKCAjw-IWkBhBTEiwA2exyO1pBoV7k3j16OANdyEOMVYDUvy4MZK3WQX6zzhymBxz7Pikqq0ndwBoCHvUQAvD_BwE&gclsrc=aw.ds#gs.2t5hw6) provides [quantization](https://intel.github.io/neural-compressor/latest/docs/source/quantization.html) to improve the speed of inference.

Check out and incorporate Intel’s other [AI and machine learning framework optimizations](https://www.intel.com/content/www/us/en/developer/tools/frameworks/overview.html) and [end-to-end portfolio of tools](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/tools.html) into your AI workflow.

Learn about the unified, open, standards-based [oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html) programming model that forms the foundation of Intel’s [AI Software Portfolio](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/overview.html) to help you prepare, build, deploy, and scale your AI solutions.

For more details about the 4th gen Intel® Xeon® Scalable processors, visit the [Intel® AI platform overview](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/platform.html) where you can learn how Intel is empowering developers to run end-to-end AI pipelines on these powerful CPUs.


## Additional Resources

* [Accelerate AI Workloads with Intel® Advanced Matrix Extensions (Intel® AMX)](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/ai-solution-brief.html)
* [AI and Machine Learning Development Tools and Resources](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/overview.html)
* [AI Frameworks](https://www.intel.com/content/www/us/en/developer/tools/frameworks/overview.html#gs.2t503z)
* [Computer Vision](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/training/computer-vision.html)
* [Intel Hardware for AI](https://www.intel.com/content/www/us/en/artificial-intelligence/hardware.html)
* [Intel Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html?cid=sem&source=sa360&campid=2023_q2_iags_us_iagsoapie_iagsoapiee_awa_text-link_exact_cd_dpd-oneapi-intel_neural_compressor_3500107853_google_div_oos_non-pbm_intel&ad_group=ai_model_compression_exact&intel_term=neural+compressor&sa360id=43700076378213630&gclid=CjwKCAjw-IWkBhBTEiwA2exyO1pBoV7k3j16OANdyEOMVYDUvy4MZK3WQX6zzhymBxz7Pikqq0ndwBoCHvUQAvD_BwE&gclsrc=aw.ds#gs.2t5hw6)
* [oneAPI Unified Programming Model](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html#gs.h7kofh)
* [PyTorch Foundation](https://pytorch.org/foundation)
* [PyTorch Optimizations from Intel](https://www.intel.com/content/www/us/en/developer/tools/oneapi/optimization-for-pytorch.html)
* [PyTorch Quantization Code Sample](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Features-and-Functionality/IntelPytorch_Quantization)
* [Quantization Using Intel Neural Compressor](https://intel.github.io/neural-compressor/latest/docs/source/quantization.html)
