---
layout: blog_detail
title: 'Model Serving in PyTorch'
author: Jeff Smith
redirect_from: /2019/05/08/model-serving-in-pyorch.html
---

PyTorch has seen a lot of adoption in research, but people can get confused about how well PyTorch models can be taken into production. This blog post is meant to clear up any confusion people might have about the road to production in PyTorch.
Usually when people talk about taking a model “to production,” they usually mean performing **inference**, sometimes called model evaluation or prediction or serving. At the level of a function call, in PyTorch, inference looks something like this:

* In Python
    * `module(input)`
* In traced modules
    * `module(input)`
* In C++
    * `at::Tensor output = module->forward(inputs).toTensor();`

Since we at Facebook perform inference operations using PyTorch hundreds of trillions of times per day, we've done a lot to make sure that inference runs as efficiently as possible.

## Serving Strategies

That zoomed-in view of how you use models in inference isn't usually the whole story, though. In a real world machine learning system, you often need to do more than just run a single inference operation in the REPL or Jupyter notebook. Instead, you usually need to integrate your model into a larger application in some way. Depending on what you need to do, you can usually take one of the following approaches.

### Direct embedding

In application settings like mobile, we often just directly call the model as part of a larger program. This isn't just for apps; usually this is how robotics and dedicated devices work as well. At a code-level, the call to the model is exactly the same as what is shown above in the section about inference shown above. A key concern is often that a Python interpreter is not present in such environments, which is why PyTorch allows you to call your models from C++ and ship a model without the need for a Python runtime.

### Model microservices

If you're using your model in a server side context and you're managing multiple models, you might choose to treat each individual model (or each individual model version) as a separate service, usually using some sort of packaging mechanism like a Docker container. Then that service is often made network accessible via some sort of service, either using JSON over HTTP or an RPC technology like gRPC. The key characteristic of this approach is that you're defining a service with a single endpoint that just calls your model. Then you do do all of your model management (promotion, rollback, etc.) via whatever system you already use to manage your services (e.g. kubernetes, ECS).

### Model servers

An additional possible solution is to use a model server. This is an application built to manage and serve models. It allows you to upload multiple models and get distinct prediction endpoints for each of them. Typically such systems include a number of other features to help solve more of the whole problem of managing and serving models. This can include things like metrics, visualization, data pre-processing, and more. Even something as simple as having a system for automatically versioning models can make building important features like model rollbacks much easier.

### Evolving Patterns

The above is a somewhat arbitrary breakdown of different approaches based on a snapshot in time. Design patterns are still evolving. Recently, model server designs have started to adopt more of the technologies of general service infrastructure such as Docker containers and kubernetes, so many model servers have started to share properties of the model microservice design discussed above. For a deeper dive into the general concepts of model server designs, you can check out my [book on machine learning systems](https://www.manning.com/books/machine-learning-systems).

## Serving PyTorch Models

So, if you're a PyTorch user, what should you use if you want to take your models to production?

If you're on mobile or working on an embedded system like a robot, direct embedding in your application is often the right choice. 
For mobile specifically, your use case might be served by the ONNX export functionality.
Note that ONNX, by its very nature, has limitations and doesn't support all of the functionality provided by the larger PyTorch project.
You can check out [this tutorial](https://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html) on deploying PyTorch models to mobile using ONNX to see if this path might suit your use case. 
That said, we've heard that there's a lot more that PyTorch users want to do on mobile, so look for more mobile-specific functionality in PyTorch in the future.
For other embedded systems, like robots, running [inference on a PyTorch model from the C++ API](https://pytorch.org/tutorials/advanced/cpp_export.html) could be the right solution.

If you can't use the cloud or prefer to manage all services using the same technology, you can follow [this example](https://medium.com/datadriveninvestor/deploy-your-pytorch-model-to-production-f69460192217) to build a simple model microservice using the Flask web framework.

If you want to manage multiple models within a non-cloud service solution, there are teams developing PyTorch support in model servers like [MLFlow](https://mlflow.org/), [Kubeflow](https://www.kubeflow.org/), and [RedisAI.](https://oss.redislabs.com/redisai/) We're excited to see innovation from multiple teams building OSS model servers, and we'll continue to highlight innovation in the PyTorch ecosystem in the future.

If you can use the cloud for your application, there are several great choices for working with models in the cloud. For AWS Sagemaker, you can start find a guide to [all of the resources from AWS for working with PyTorch](https://docs.aws.amazon.com/sagemaker/latest/dg/pytorch.html), including docs on how to use the [Sagemaker Python SDK](https://sagemaker.readthedocs.io/en/stable/using_pytorch.html). You can also see [some](https://youtu.be/5h1Ot2dPi2E) [talks](https://youtu.be/qc5ZikKw9_w) we've given on using PyTorch on Sagemaker. Finally, if you happen to be using PyTorch via FastAI, then they've written a really simple guide to getting up and running on Sagemaker. 

The story is similar across other major clouds. On Google Cloud, you can follow [these instructions](https://cloud.google.com/deep-learning-vm/docs/pytorch_start_instance) to get access to a Deep Learning VM with PyTorch pre-installed. On Microsoft Azure, you have a number of ways to get started from [Azure Machine Learning Service](https://azure.microsoft.com/en-us/services/machine-learning-service/) to [Azure Notebooks](https://notebooks.azure.com/pytorch/projects/tutorials) showing how to use PyTorch.

## Your Models

Whichever approach you take to bringing your PyTorch models to production, we want to support you and enable your success. Do you love one of the options above? Are you having difficulty with that one crucial feature you can't find support for? We'd love to discuss more on the [deployment category](https://discuss.pytorch.org/c/deployment) on the PyTorch Discuss forums. We'd love to help, and where you're seeing success, amplify your story.
