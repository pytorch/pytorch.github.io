---
layout: blog_detail
title: 'Updates & Improvements to PyTorch Tutorials'
author: Team PyTorch
---

PyTorch.org provides researchers and developers with documentation, installation instructions, latest news, community projects, tutorials, and more. Today, we are introducing usability and content improvements including tutorials in additional categories, a new recipe format for quickly referencing common topics, sorting using tags, and an updated homepage. 

Let’s take a look at them in detail. 

## TUTORIALS HOME PAGE UPDATE
The tutorials home page now provides clear actions that developers can take. For new PyTorch users, there is an easy-to-discover button to take them directly to “A 60 Minute Blitz”. Right next to it, there is a button to view all recipes which are designed to teach specific features quickly with examples. 

<div class="text-center">
  <img src="{{ site.url }}/assets/images/tutorialhomepage.png" width="100%">
</div>

In addition to the existing left navigation bar, tutorials can now be quickly filtered by multi-select tags. Let’s say you want to view all tutorials related to “Production” and “Quantization”. You can select the “Production” and “Quantization” filters as shown in the image shown below:

<div class="text-center">
  <img src="{{ site.url }}/assets/images/blockfiltering.png" width="100%">
</div>

The following additional resources can also be found at the bottom of the Tutorials homepage:
* [PyTorch Cheat Sheet](https://pytorch.org/tutorials/beginner/ptcheat.html)
* [PyTorch Examples](https://github.com/pytorch/examples)
* [Tutorial on GitHub](https://github.com/pytorch/tutorials)

## PYTORCH RECIPES  
Recipes are new bite-sized, actionable examples designed to teach researchers and developers how to use specific PyTorch features. Some notable new recipes include:
* [Loading Data in PyTorch](https://pytorch.org/tutorials/recipes/recipes/loading_data_recipe.html)
* [Model Interpretability Using Captum](https://pytorch.org/tutorials/recipes/recipes/Captum_Recipe.html)
* [How to Use TensorBoard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)

View the full recipes [here](http://pytorch.org/tutorials/recipes/recipes_index.html).

## LEARNING PYTORCH
This section includes tutorials designed for users new to PyTorch. Based on community feedback, we have made updates to the current [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) tutorial, one of our most popular tutorials for beginners. Upon completion, one can understand what PyTorch and neural networks are, and be able to build and train a simple image classification network. Updates include adding explanations to clarify output meanings and linking back to where users can read more in the docs, cleaning up confusing syntax errors, and reconstructing and explaining new concepts for easier readability.  

## DEPLOYING MODELS IN PRODUCTION
This section includes tutorials for developers looking to take their PyTorch models to production. The tutorials include:
* [Deploying PyTorch in Python via a REST API with Flask](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)
* [Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
* [Loading a TorchScript Model in C++](https://pytorch.org/tutorials/advanced/cpp_export.html)
* [Exporting a Model from PyTorch to ONNX and Running it using ONNX Runtime](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)

## FRONTEND APIS
PyTorch provides a number of frontend API features that can help developers to code, debug, and validate their models more efficiently. This section includes tutorials that teach what these features are and how to use them. Some tutorials to highlight: 
* [Introduction to Named Tensors in PyTorch](https://pytorch.org/tutorials/intermediate/named_tensor_tutorial.html)
* [Using the PyTorch C++ Frontend](https://pytorch.org/tutorials/advanced/cpp_frontend.html)
* [Extending TorchScript with Custom C++ Operators](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html)
* [Extending TorchScript with Custom C++ Classes](https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html)
* [Autograd in C++ Frontend](https://pytorch.org/tutorials/advanced/cpp_autograd.html)

## MODEL OPTIMIZATION
Deep learning models often consume large amounts of memory, power, and compute due to their complexity. This section provides tutorials for model optimization:
* [Pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
* [Dynamic Quantization on BERT](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html)
* [Static Quantization with Eager Mode in PyTorch](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)

## PARALLEL AND DISTRIBUTED TRAINING
PyTorch provides features that can accelerate performance in research and production such as native support for asynchronous execution of collective operations and peer-to-peer communication that is accessible from Python and C++. This section includes tutorials on parallel and distributed training: 
* [Single-Machine Model Parallel Best Practices](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)
* [Getting started with Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
* [Getting started with Distributed RPC Framework](https://pytorch.org/tutorials/intermediate/rpc_tutorial.html)
* [Implementing a Parameter Server Using Distributed RPC Framework](https://pytorch.org/tutorials/intermediate/rpc_param_server_tutorial.html)

Making these improvements are just the first step of improving PyTorch.org for the community. Please submit your suggestions [here](https://github.com/pytorch/tutorials/pulls).

Cheers,

Team PyTorch
