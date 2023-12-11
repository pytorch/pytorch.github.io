---
layout: blog_detail
title: 'PyTorch library updates including new model serving library '
author: Team PyTorch
---


Along with the PyTorch 1.5 release, we are announcing new libraries for high-performance PyTorch model serving and tight integration with TorchElastic and Kubernetes. Additionally, we are releasing updated packages for torch_xla (Google Cloud TPUs), torchaudio, torchvision, and torchtext. All of these new libraries and enhanced capabilities are available today and accompany all of the core features [released in PyTorch 1.5](https://pytorch.org/blog/pytorch-1-dot-5-released-with-new-and-updated-apis). 

## TorchServe (Experimental)

TorchServe is a flexible and easy to use library for serving PyTorch models in production performantly at scale. It is cloud and environment agnostic and supports features such as multi-model serving, logging, metrics, and the creation of RESTful endpoints for application integration. TorchServe was jointly developed by engineers from Facebook and AWS with feedback and engagement from the broader PyTorch community. The experimental release of TorchServe is available today. Some of the highlights include:

* Support for both Python-based and TorchScript-based models
* Default handlers for common use cases (e.g., image segmentation, text classification) as well as the ability to write custom handlers for other use cases
* Model versioning, the ability to run multiple versions of a model at the same time, and the ability to roll back to an earlier version
* The ability to package a model, learning weights, and supporting files (e.g., class mappings, vocabularies) into a single, persistent artifact (a.k.a. the “model archive”)
* Robust management capability, allowing full configuration of models, versions, and individual worker threads via command line, config file, or run-time API
* Automatic batching of individual inferences across HTTP requests
* Logging including common metrics, and the ability to incorporate custom metrics
* Ready-made Dockerfile for easy deployment
* HTTPS support for secure deployment

To learn more about the APIs and the design of this feature, see the links below:
* See <here> for a full multi-node deployment reference architecture.
* The full documentation can be found [here](https://pytorch.org/serve).

## TorchElastic integration with Kubernetes (Experimental)

[TorchElastic](https://github.com/pytorch/elastic) is a proven library for training large scale deep neural networks at scale within companies like Facebook, where having the ability to dynamically adapt to server availability and scale as new compute resources come online is critical. Kubernetes enables customers using machine learning frameworks like PyTorch to run training jobs distributed across fleets of powerful GPU instances like the Amazon EC2 P3. Distributed training jobs, however, are not fault-tolerant, and a job cannot continue if a node failure or reclamation interrupts training. Further, jobs cannot start without acquiring all required resources, or scale up and down without being restarted. This lack of resiliency and flexibility results in increased training time and costs from idle resources. TorchElastic addresses these limitations by enabling distributed training jobs to be executed in a fault-tolerant and elastic manner. Until today, Kubernetes users needed to manage Pods and Services required for TorchElastic training jobs manually.

Through the joint collaboration of engineers at Facebook and AWS, TorchElastic, adding elasticity and fault tolerance, is now supported using vanilla Kubernetes and through the managed EKS service from AWS.

To learn more see the [TorchElastic repo](http://pytorch.org/elastic/0.2.0rc0/kubernetes.html) for the controller implementation and docs on how to use it.

## torch_xla 1.5 now available

[torch_xla](http://pytorch.org/xla/) is a Python package that uses the [XLA linear algebra compiler](https://www.tensorflow.org/xla) to accelerate the [PyTorch deep learning framework](https://pytorch.org/) on [Cloud TPUs](https://cloud.google.com/tpu/) and [Cloud TPU Pods](https://cloud.google.com/tpu/docs/tutorials/pytorch-pod). torch_xla aims to give PyTorch users the ability to do everything they can do on GPUs on Cloud TPUs as well while minimizing changes to the user experience. The project began with a conversation at NeurIPS 2017 and gathered momentum in 2018 when teams from Facebook and Google came together to create a proof of concept. We announced this collaboration at PTDC 2018 and made the PyTorch/XLA integration broadly available at PTDC 2019. The project already has 28 contributors, nearly 2k commits, and a repo that has been forked more than 100 times. 

This release of [torch_xla](http://pytorch.org/xla/) is aligned and tested with PyTorch 1.5 to reduce friction for developers and to provide a stable and mature PyTorch/XLA stack for training models using Cloud TPU hardware. You can [try it for free](https://medium.com/pytorch/get-started-with-pytorch-cloud-tpus-and-colab-a24757b8f7fc) in your browser on an 8-core Cloud TPU device with [Google Colab](https://colab.research.google.com/), and you can use it at a much larger scaleon [Google Cloud](https://cloud.google.com/gcp).

See the full torch_xla release notes [here](https://github.com/pytorch/xla/releases). Full docs and tutorials can be found [here](https://pytorch.org/xla/) and [here](https://cloud.google.com/tpu/docs/tutorials).

## PyTorch Domain Libraries

torchaudio, torchvision, and torchtext complement PyTorch with common datasets, models, and transforms in each domain area. We’re excited to share new releases for all three domain libraries alongside PyTorch 1.5 and the rest of the library updates. For this release, all three domain libraries are removing support for Python2 and will support Python3 only.

### torchaudio 0.5
The torchaudio 0.5 release includes new transforms, functionals, and datasets. Highlights for the release include:

* Added the Griffin-Lim functional and transform, `InverseMelScale` and `Vol` transforms, and `DB_to_amplitude`. 
* Added support for `allpass`, `fade`, `bandpass`, `bandreject`, `band`, `treble`, `deemph`, and `riaa` filters and transformations.
* New datasets added including `LJSpeech` and `SpeechCommands` datasets. 

See the release full notes [here](https://github.com/pytorch/audio/releases) and full docs can be found [here](https://pytorch.org/audio/).

### torchvision 0.6
The torchvision 0.6 release includes updates to datasets, models and a significant number of bug fixes. Highlights include:

* Faster R-CNN now supports negative samples which allows the feeding of images without annotations at training time.
* Added `aligned` flag to `RoIAlign` to match Detectron2. 
* Refactored abstractions for C++ video decoder

See the release full notes [here](https://github.com/pytorch/vision/releases) and full docs can be found [here](https://pytorch.org/vision/stable/index.html).

### torchtext 0.6
The torchtext 0.6 release includes a number of bug fixes and improvements to documentation. Based on user's feedback, dataset abstractions are currently being redesigned also. Highlights for the release include:

* Fixed an issue related to the SentencePiece dependency in conda package.
* Added support for the experimental IMDB dataset to allow a custom vocab.
* A number of documentation updates including adding a code of conduct and a deduplication of the docs on the torchtext site. 

Your feedback and discussions on the experimental datasets API are welcomed. You can send them to [issue #664](https://github.com/pytorch/text/issues/664). We would also like to highlight the pull request [here](https://github.com/pytorch/text/pull/701) where the latest dataset abstraction is applied to the text classification datasets. The feedback can be beneficial to finalizing this abstraction. 

See the release full notes [here](https://github.com/pytorch/text/releases) and full docs can be found [here](https://pytorch.org/text/).


*We’d like to thank the entire PyTorch team, the Amazon team and the community for all their contributions to this work.*

Cheers!

Team PyTorch
