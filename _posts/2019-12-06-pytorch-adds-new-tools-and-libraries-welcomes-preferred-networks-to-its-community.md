---
layout: blog_detail
title: 'PyTorch adds new tools and libraries, welcomes Preferred Networks to its community'
author: Team PyTorch
---

PyTorch continues to be used for the latest state-of-the-art research on display at the NeurIPS conference next week, making up nearly [70% of papers](https://chillee.github.io/pytorch-vs-tensorflow/) that cite a framework. In addition, we’re excited to welcome Preferred Networks, the maintainers of the Chainer framework, to the PyTorch community. Their teams are moving fully over to PyTorch for developing their ML capabilities and services.

This growth underpins PyTorch’s focus on building for the needs of the research community, and increasingly, supporting the full workflow from research to production deployment. To further support researchers and developers, we’re launching a number of new tools and libraries for large scale computer vision and elastic fault tolerant training. Learn more on GitHub and at our NeurIPS booth.

## Preferred Networks joins the PyTorch community

Preferred Networks, Inc. (PFN) announced plans to move its deep learning framework from Chainer to PyTorch. As part of this change, PFN will collaborate with the PyTorch community and contributors, including people from Facebook, Microsoft, CMU, and NYU, to participate in the development of PyTorch.

PFN developed Chainer, a deep learning framework that introduced the concept of define-by-run (also referred to as eager execution), to support and speed up its deep learning development. Chainer has been used at PFN since 2015 to rapidly solve real-world problems with the latest, cutting-edge technology. Chainer was also one of the inspirations for PyTorch’s initial design, as outlined in the [PyTorch NeurIPS](https://papers.nips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library) paper.

PFN has driven innovative work with [CuPy](https://cupy.chainer.org/), ImageNet in 15 minutes, [Optuna](https://optuna.org/), and other projects that have pushed the boundaries of design and engineering. As part of the PyTorch community, PFN brings with them creative engineering capabilities and experience to help take the framework forward. In addition, PFN’s migration to PyTorch will allow it to efficiently incorporate the latest research results to accelerate its R&D activities, [given PyTorch’s broad adoption with researchers](https://thegradient.pub/state-of-ml-frameworks-2019-pytorch-dominates-research-tensorflow-dominates-industry/), and to collaborate with the community to add support for PyTorch on MN-Core, a deep learning processor currently in development.

We are excited to welcome PFN to the PyTorch community, and to jointly work towards the common goal of furthering advances in deep learning technology. Learn more about the PFN’s migration to PyTorch [here](https://preferred.jp/en/news/pr20191205/).

## Tools for elastic training and large scale computer vision

### PyTorch Elastic (Experimental)

Large scale model training is becoming commonplace with architectures like BERT and the growth of model parameter counts into the billions or even tens of billions. To achieve convergence at this scale in a reasonable amount of time, the use of distributed training is needed.

The current PyTorch Distributed Data Parallel (DDP) module enables data parallel training where each process trains the same model but on different shards of data. It enables bulk synchronous, multi-host, multi-GPU/CPU execution of ML training. However, DDP has several shortcomings; e.g. jobs cannot start without acquiring all the requested nodes; jobs cannot continue after a node fails due to error or transient issue; jobs cannot incorporate a node that joined later; and lastly; progress cannot be made with the presence of a slow/stuck node.

The focus of [PyTorch Elastic](https://github.com/pytorch/elastic), which uses Elastic Distributed Data Parallelism, is to address these issues and build a generic framework/APIs for PyTorch to enable reliable and elastic execution of these data parallel training workloads. It will provide better programmability, higher resilience to failures of all kinds, higher-efficiency and larger-scale training compared with pure DDP.

Elasticity, in this case, means both: 1) the ability for a job to continue after node failure (by running with fewer nodes and/or by incorporating a new host and transferring state to it); and 2) the ability to add/remove nodes dynamically due to resource availability changes or bottlenecks.

While this feature is still experimental, you can try it out on AWS EC2, with the instructions [here](https://github.com/pytorch/elastic/tree/master/aws). Additionally, the PyTorch distributed team is working closely with teams across AWS to support PyTorch Elastic training within services such as Amazon Sagemaker and Elastic Kubernetes Service (EKS). Look for additional updates in the near future.

### New Classification Framework

Image and video classification are at the core of content understanding. To that end, you can now leverage a new end-to-end framework for large-scale training of state-of-the-art image and video classification models. It allows researchers to quickly prototype and iterate on large distributed training jobs at the scale of billions of images. Advantages include:

* Ease of use - This framework features a modular, flexible design that allows anyone to train machine learning models on top of PyTorch using very simple abstractions. The system also has out-of-the-box integration with AWS on PyTorch Elastic, facilitating research at scale and making it simple to move between research and production.
* High performance - Researchers can use the framework to train models such as Resnet50 on ImageNet in as little as 15 minutes.

You can learn more at the NeurIPS Expo workshop on Multi-Modal research to production or get started with the PyTorch Elastic Imagenet example [here](https://github.com/pytorch/elastic/blob/master/examples/imagenet/main.py).

## Come see us at NeurIPS

The PyTorch team will be hosting workshops at NeurIPS during the industry expo on 12/8. Join the sessions below to learn more, and visit the team at the PyTorch booth on the show floor and during the Poster Session. At the booth, we’ll be walking through an interactive demo of PyTorch running fast neural style transfer on a Cloud TPU - here’s a [sneak peek](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/style_transfer_inference-xrt-1-15.ipynb).

We’re also publishing a [paper that details the principles that drove the implementation of PyTorch](https://papers.nips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library) and how they’re reflected in its architecture.

*Multi-modal Research to Production* - This workshop will dive into a number of modalities such as computer vision (large scale image classification and instance segmentation) and Translation and Speech (seq-to-seq Transformers) from the lens of taking cutting edge research to production. Lastly, we will also walk through how to use the latest APIs in PyTorch to take eager mode developed models into graph mode via Torchscript and quantize them for scale production deployment on servers or mobile devices. Libraries used include:

* Classification Framework - a newly open sourced PyTorch framework developed by Facebook AI for research on large-scale image and video classification. It allows researchers to quickly prototype and iterate on large distributed training jobs. Models built on the framework can be seamlessly deployed to production.
* Detectron2 - the recently released object detection library built by the Facebook AI Research computer vision team. We will articulate the improvements over the previous version including: 1) Support for latest models and new tasks; 2) Increased flexibility, to enable new computer vision research; 3) Maintainable and scalable, to support production use cases.
* Fairseq - general purpose sequence-to-sequence library, can be used in many applications, including (unsupervised) translation, summarization, dialog and speech recognition.

*Responsible and Reproducible AI* - This workshop on Responsible and Reproducible AI will dive into important areas that are shaping the future of how we interpret, reproduce research, and build AI with privacy in mind. We will cover major challenges, walk through solutions, and finish each talk with a hands-on tutorial.

* Reproducibility: As the number of research papers submitted to arXiv and conferences skyrockets, scaling reproducibility becomes difficult. We must address the following challenges: aid extensibility by standardizing code bases, democratize paper implementation by writing hardware agnostic code, facilitate results validation by documenting “tricks” authors use to make their complex systems function. To offer solutions, we will dive into tool like PyTorch Hub and PyTorch Lightning which are used by some of the top researchers in the world to reproduce the state of the art.
* Interpretability: With the increase in model complexity and the resulting lack of transparency, model interpretability methods have become increasingly important. Model understanding is both an active area of research as well as an area of focus for practical applications across industries using machine learning. To get hands on, we will use the recently released Captum library that provides state-of-the-art algorithms to provide researchers and developers with an easy way to understand the importance of neurons/layers and the predictions made by our models.`
* Private AI: Practical applications of ML via cloud-based or machine-learning-as-a-service platforms pose a range of security and privacy challenges. There are a number of technical approaches being studied including: homomorphic encryption, secure multi-party computation, trusted execution environments, on-device computation, and differential privacy. To provide an immersive understanding of how some of these technologies are applied, we will use the CrypTen project which provides a community based research platform to take the field of Private AI forward.

*We’d like to thank the entire PyTorch team and the community for all their contributions to this work.*

Cheers!

Team PyTorch
