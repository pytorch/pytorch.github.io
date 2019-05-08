---
layout: blog_detail
title: 'PyTorch adds new dev tools as it hits production scale'
author: The PyTorch Team
---

_This is a partial re-post of the original blog post on the Facebook AI Blog. The full post can be [viewed here](https://ai.facebook.com/blog/pytorch-adds-new-dev-tools-as-it-hits-production-scale/)_

Since its release just a few months ago, [PyTorch 1.0](http://pytorch.org/) has been rapidly adopted as a powerful, flexible deep learning platform that enables engineers and researchers to move quickly from research to production. We are highlighting some of the ways the AI engineering and research community is using PyTorch 1.0. We’re also sharing new details about the latest release, PyTorch 1.1, and showcasing some of the new development tools created by the community.

Building on the initial launch of PyTorch in 2017, we partnered with the AI community to ship the stable release of PyTorch 1.0 [last December](https://code.fb.com/ai-research/pytorch-developer-ecosystem-expands-1-0-stable-release/). Along with enhanced production-oriented capabilities and deep integration with leading cloud platforms, PyTorch 1.0 expands on the open source library’s core features, with the addition of PyTorch JIT (Just in time compilation) that seamlessly transitions between eager mode and graph mode to provide both flexibility and speed.

Leading businesses across industries are beginning to use PyTorch to both facilitate their research and then also deploy at large scale for applications such as translation, computer vision, conversational interfaces, pharmaceutical research, factory optimization, and automated driving research. Community adoption of PyTorch has also continued to expand. Stanford, UC Berkeley, Caltech, and other universities are using PyTorch as a fundamental tool for their machine learning (ML) courses; new ecosystem projects have launched to support development on PyTorch; and major cloud platforms have expanded their integration with PyTorch.

## Using PyTorch across industries

Many leading businesses are moving to PyTorch 1.0 to accelerate development and deployment of new AI systems. Here are some examples:

- Airbnb leveraged PyTorch's rich libraries and APIs for conversational AI and deployed a Smart Reply to help the company’s service agents respond more effectively to customers.
- [ATOM](https://atomscience.org/) is building a platform to generate and optimize new drug candidates significantly faster and with greater success than conventional processes. Using machine learning frameworks such as PyTorch, ATOM was able to design a variational autoencoder for representing diverse chemical structures and designing new drug candidates.
- Genentech is utilizing PyTorch’s flexible control structures and dynamic graphs to train deep learning models that will aid in the development of individualized cancer therapy.
- Microsoft is using PyTorch across its organization to develop ML models at scale and deploy them via the ONNX Runtime. Using PyTorch, Microsoft Cognition has built distributed language models that scale to billions of words and are now in production in offerings such as Cognitive Services.
- Toyota Research Institute (TRI) is developing a two-pronged approach toward automated driving with Toyota Guardian and Toyota Chauffeur technologies. The Machine Learning Team at TRI is creating new deep learning algorithms to leverage Toyota's 10 million sales per year data advantage. The flexibility of PyTorch has vastly accelerated their pace of exploration and its new production features will enable faster deployment towards their safety critical applications.

Following the release of PyTorch 1.0 in December 2018, we’re now announcing the availability of v1.1, which improves performance, adds new model understanding and visualization tools to improve usability, and provides new APIs.

Key features of PyTorch v1.1 include:

- [TensorBoard](https://www.tensorflow.org/tensorboard): First-class and native support for visualization and model debugging with TensorBoard, a web application suite for inspecting and understanding training runs and graphs. PyTorch now natively supports TensorBoard with a simple “from torch.utils.tensorboard import SummaryWriter” command.
- JIT compiler: Improvements to just-in-time (JIT) compilation. These include various bug fixes as well as expanded capabilities in TorchScript, such as support for dictionaries, user classes, and attributes.
- New APIs: Support for Boolean tensors and better support for custom recurrent neural networks.
- Distributed Training: Improved performance for common models such as CNNs, added support for multi device modules including the ability to split models across GPUs while still using Distributed Data Parallel (DDP) and support for modules where not all parameters are used in every iteration (e.g. control flow, like adaptive softmax, etc). See the latest tutorials [here](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html).

We’ve also continued to partner with the community to foster projects and tools aimed at supporting ML engineers for needs ranging from improved model understanding to auto-tuning using AutoML methods. With the release of Ax and BoTorch (below), we will be sharing some of our core algorithms, including meta-learning for efficiently optimizing hyperparameters from based on historical tasks. We are excited to see this work open-sourced for the community to build on.

This ecosystem includes open source projects and tools that have been deployed at production scale, as well as products and services from our partnership with industry leaders who share our vision of an open and collaborative AI community. Here are a few of the latest tools:

- [BoTorch](https://ai.facebook.com/blog/open-sourcing-ax-and-botorch-new-ai-tools-for-adaptive-experimentation/): BoTorch is a research framework built on top of PyTorch to provide Bayesian optimization, a sample-efficient technique for sequential optimization of costly-to-evaluate black-box functions.
- [Ax](https://ai.facebook.com/blog/open-sourcing-ax-and-botorch-new-ai-tools-for-adaptive-experimentation/): Ax is an ML platform for managing adaptive experiments. It enables researchers and engineers to systematically explore large configuration spaces in order to optimize machine learning models, infrastructure, and products.
- [PyTorch-BigGraph](https://ai.facebook.com/blog/open-sourcing-pytorch-biggraph-for-faster-embeddings-of-extremely-large-graphs/): PBG is a distributed system for creating embeddings of very large graphs with billions of entities and trillions of edges. It includes support for sharding and negative sampling and it offers sample use cases based on Wikidata embeddings.
- [Google AI Platform Notebooks](https://cloud.google.com/ai-platform-notebooks/): AI Platform Notebooks is a new, hosted JupyterLab service from Google Cloud Platform. Data scientists can quickly create virtual machines running JupyterLab with the latest version of PyTorch preinstalled. It is also tightly integrated with GCP services such as BigQuery, Cloud Dataproc, Cloud Dataflow, and AI Factory, making it easy to execute the full ML cycle without ever leaving JupyterLab.

We’re also excited to see many interesting new projects from the broader PyTorch community. Highlights include:

- [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch):This is a full PyTorch reimplementation that uses gradient accumulation to provide the benefits of big batches on as few as four GPUs.
- [GeomLoss](http://www.kernel-operations.io/geomloss/index.html): A Python API that defines PyTorch layers for geometric loss functions between sampled measures, images, and volumes. It includes MMD, Wasserstein, Sinkhorn, and more.

<div class="text-center">
  <img src="{{ site.url }}/assets/images/geomloss.jpg" width="100%">
</div>

- [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric): A deep learning extension library for PyTorch that offers several methods for deep learning on graphs and other irregular structures (also known as [geometric deep learning](http://geometricdeeplearning.com)) from a variety of published papers.
- [Curve-GCN](https://github.com/fidler-lab/curve-gcn): A real-time, interactive image annotation approach that uses an end-to-end-trained graph convolutional network (GCN). It supports object annotation by either polygons or splines, facilitating labeling efficiency for both line-based and curved objects. Curve-GCN runs 10x faster than traditional methods, such as Polygon-RNN++.

## Udacity, fast.ai, and others develop new PyTorch resources

PyTorch is ideal for teaching ML development because it enables rapid experimentation through its flexible, dynamic programming environment and user-friendly Pythonic interface. In addition, Google Colab now offers an interactive Jupyter Notebook environment that natively supports PyTorch, allowing developers to run any PyTorch tutorial immediately with free CPU and GPU resources.

University-level classes — including [Stanford NLP](http://web.stanford.edu/class/cs224n), [UC Berkeley](https://inst.eecs.berkeley.edu/~cs280/sp18/) Computer Vision, and [Caltech](http://cast.caltech.edu) Robotics courses — are now being taught on PyTorch. In addition, massive open online courses (MOOCs) are training thousands of new PyTorch developers.

Today, we’re announcing a [new Udacity course](https://blog.udacity.com/2019/05/announcing-the-secure-and-private-ai-scholarship-challenge-with-facebook.html), building upon the Intro to Deep Learning course launched last year. This new course, led by Andrew Trask of Oxford University and OpenMined, covers important concepts around privacy in AI, including methods such as differential privacy and federated learning. Facebook will also be providing scholarships to support students as they continue their ML education in Udacity’s full Nanodegree programs.

The [fast.ai](https://www.fast.ai) community is also continuing to invest energy and resources in PyTorch. In June, fast.ai will launch a new course called Deep Learning from the Foundations, which will show developers how to go all the way from writing matrix multiplication from scratch to how to train and implement a state-of-the-art ImageNet model. The course will include deep dives into the underlying implementation of methods in the PyTorch and fast.ai libraries, and will use the code to explain and illustrate the academic papers that underlie these methods.

As part of the course, fast.ai will also release new software modules, including fastai.audio, which brings the power of fast.ai’s deep abstractions and curated algorithms to the new PyTorch.audio module, and show how fastai.vision can be used to [create stunning high-resolution videos](https://www.fast.ai/2019/05/03/decrappify) from material such as old classic movies, and from cutting-edge microscopy sequences through a collaboration with the [Salk Institute](https://www.salk.edu). In addition, fast.ai is contributing its new X-ResNet module, including a suite of models pretrained on ImageNet.

## Getting started with PyTorch

Everyone in the AI community — including those new to ML development as well as researchers and engineers looking for ways to accelerate their end-to-end workflows — can experiment with PyTorch instantly by visiting [pytorch.org](https://pytorch.org) and launching a [tutorial](https://pytorch.org/tutorials) in Colab. There are also many easy ways to [get started](https://pytorch.org/get-started/locally) both locally and on popular cloud platforms.
