---
layout: blog_detail
title: "Easily list and initialize models with new APIs in TorchVision"
author: Vasilis Vryniotis and Laurence Rouesnel
featured-img: "/assets/images/easily-list-and-initialize-models-with-new-apis-in-torchvision-1.png"
---

TorchVision now supports listing and initializing all available built-in models and weights by name. This new API builds upon the recently introduced [Multi-weight support API](https://pytorch.org/blog/introducing-torchvision-new-multi-weight-support-api/), is currently in Beta, and it addresses a long-standing [request](https://github.com/pytorch/vision/issues/1143) from the community.

<p align="center">
  <img src="\assets\images\easily-list-and-initialize-models-with-new-apis-in-torchvision.gif" width="100%">
</p>

You can try out the new API in the [latest nightly](https://pytorch.org/get-started/locally/) release of TorchVision. We’re looking to collect feedback ahead of finalizing the feature in TorchVision v0.14. We have created a dedicated [Github Issue](https://github.com/pytorch/vision/issues/6365) where you can post your comments, questions and suggestions!

## Querying and initializing available models

Before the new model registration API, developers had to query the ``__dict__`` attribute of the modules in order to list all available models or to fetch a specific model builder method by its name:

```python
# Initialize a model by its name:
model = torchvision.models.__dict__[model_name]()

# List available models:
available_models = [
    k for k, v in torchvision.models.__dict__.items()
    if callable(v) and k[0].islower() and k[0] != "_"
]
```

The above approach does not always produce the expected results and is hard to discover. For example, since the [``get_weight()``](https://pytorch.org/vision/main/models.html#using-models-from-hub) method is exposed publicly under the same module, it will be included in the list despite not being a model. In general, reducing the verbosity (less imports, shorter names etc) and being able to initialize models and weights directly from their names (better support of configs, TorchHub etc) was [feedback](https://github.com/pytorch/vision/issues/5088) provided previously by the community. To solve this problem, we have developed a model registration API.

## A new approach

We’ve added 4 new methods under the torchvision.models module:

```python
from torchvision.models import get_model, get_model_weights, get_weight, list_models
```

The styles and naming conventions align closely with a prototype mechanism proposed by Philip Meier for the [Datasets V2](https://github.com/pytorch/vision/blob/main/torchvision/prototype/datasets/_api.py) API, aiming to offer a similar user experience. The model registration methods are kept private on purpose as we currently focus only on supporting the built-in models of TorchVision.

### List models

Listing all available models in TorchVision can be done with a single function call:

```python
>>> list_models()
['alexnet', 'mobilenet_v3_large', 'mobilenet_v3_small', 'quantized_mobilenet_v3_large', ...]
```

To list the available models of specific submodules:

```python
>>> list_models(module=torchvision.models)
['alexnet', 'mobilenet_v3_large', 'mobilenet_v3_small', ...]
>>> list_models(module=torchvision.models.quantization)
['quantized_mobilenet_v3_large', ...]
```

### Initialize models

Now that you know which models are available, you can easily initialize a model with pre-trained weights:

```python
>>> get_model("quantized_mobilenet_v3_large", weights="DEFAULT")
QuantizableMobileNetV3(
  (features): Sequential(
   ....
   )
)
```

### Get weights
Sometimes, while working with config files or using TorchHub, you might have the name of a specific weight entry and wish to get its instance. This can be easily done with the following method:

```python
>>> get_weight("ResNet50_Weights.IMAGENET1K_V2")
ResNet50_Weights.IMAGENET1K_V2
```

To get the enum class with all available weights of a specific model you can use either its name:

```python
>>> get_model_weights("quantized_mobilenet_v3_large")
<enum 'MobileNet_V3_Large_QuantizedWeights'>
```

Or its model builder method:

```python
>>> get_model_weights(torchvision.models.quantization.mobilenet_v3_large)
<enum 'MobileNet_V3_Large_QuantizedWeights'>
```

### TorchHub support
The new methods are also available via TorchHub:

```python
import torch

# Fetching a specific weight entry by its name:
weights = torch.hub.load("pytorch/vision", "get_weight", weights="ResNet50_Weights.IMAGENET1K_V2")

# Fetching the weights enum class to list all available entries:
weight_enum = torch.hub.load("pytorch/vision", "get_model_weights", name="resnet50")
print([weight for weight in weight_enum])
```

## Putting it all together

For example, if you wanted to retrieve all the small-sized models with pre-trained weights and initialize one of them, it’s a matter of using the above APIs:

```python
import torchvision
from torchvision.models import get_model, get_model_weights, list_models


max_params = 5000000

tiny_models = []
for model_name in list_models(module=torchvision.models):
    weights_enum = get_model_weights(model_name)
    if len([w for w in weights_enum if w.meta["num_params"] <= max_params]) > 0:
        tiny_models.append(model_name)

print(tiny_models)
# ['mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mobilenet_v2', ...]

model = get_model(tiny_models[0], weights="DEFAULT")
print(sum(x.numel() for x in model.state_dict().values()))
# 2239188
```

For more technical details please see the original [RFC](https://github.com/pytorch/vision/pull/6330). Please spare a few minutes to provide your feedback on the new API, as this is crucial for graduating it from beta and including it in the next release. You can do this on the dedicated [Github Issue](https://github.com/pytorch/vision/issues/6365). We are looking forward to reading your comments!