---
layout: blog_detail
title: 'OpenMined and PyTorch partner to launch fellowship funding for privacy-preserving ML community'
author: Andrew Trask (OpenMined/U.Oxford), Shubho Sengupta, Laurens van der Maaten, Joe Spisak
excerpt: Many applications of machine learning (ML) pose a range of security and privacy challenges.
---

<div class="text-center">
  <img src="{{ site.url }}/assets/images/openmined-pytorch.png" width="100%">
</div>

Many applications of machine learning (ML) pose a range of security and privacy challenges. In particular, users may not be willing or allowed to share their data, which prevents them from taking full advantage of ML platforms like PyTorch. To take the field of privacy-preserving ML (PPML) forward, OpenMined and PyTorch are announcing plans to jointly develop a combined platform to accelerate PPML research as well as new funding for fellowships.

There are many techniques attempting to solve the problem of privacy in ML, each at various levels of maturity. These include (1) homomorphic encryption, (2) secure multi-party computation, (3) trusted execution environments, (4) on-device computation, (5) federated learning with secure aggregation, and (6) differential privacy. Additionally, a number of open source projects implementing these techniques were created with the goal of enabling research at the intersection of privacy, security, and ML. Among them, PySyft and CrypTen have taken an “ML-first” approach by presenting an API that is familiar to the ML community, while masking the complexities of privacy and security protocols. We are excited to announce that these two projects are now collaborating closely to build a mature PPML ecosystem around PyTorch.

Additionally, to bolster this ecosystem and take the field of privacy preserving ML forward, we are also calling for contributions and supporting research efforts on this combined platform by providing funding to support the OpenMined community and the researchers that contribute, build proofs of concepts and desire to be on the cutting edge of how privacy-preserving technology is applied. We will provide funding through the [RAAIS Foundation](https://www.raais.org/), a non-profit organization with a mission to advance education and research in artificial intelligence for the common good. We encourage interested parties to apply to one or more of the fellowships listed below.

## Tools Powering the Future of Privacy-Preserving ML

The next generation of privacy-preserving open source tools enable ML researchers to easily experiment with ML models using secure computing techniques without needing to be cryptography experts. By integrating with PyTorch, PySyft and CrypTen offer familiar environments for ML developers to research and apply these techniques as part of their work.

**PySyft** is a Python library for secure and private ML developed by the OpenMined community. It is a flexible, easy-to-use library that makes secure computation techniques like [multi-party computation (MPC)](https://en.wikipedia.org/wiki/Secure_multi-party_computation) and privacy-preserving techniques like [differential privacy](https://en.wikipedia.org/wiki/Differential_privacy) accessible to the ML community. It prioritizes ease of use and focuses on integrating these techniques into end-user use cases like federated learning with mobile phones and other edge devices, encrypted ML as a service, and privacy-preserving data science.

**CrypTen** is a framework built on PyTorch that enables private and secure ML for the PyTorch community. It is the first step along the journey towards a privacy-preserving mode in PyTorch that will make secure computing techniques accessible beyond cryptography researchers. It currently implements [secure multiparty computation](https://en.wikipedia.org/wiki/Secure_multi-party_computation) with the goal of offering other secure computing backends in the near future. Other benefits to ML researchers include:

* It is **ML first** and presents secure computing techniques via a CrypTensor object that looks and feels exactly like a PyTorch Tensor. This allows the user to use automatic differentiation and neural network modules akin to those in PyTorch.
* The framework focuses on **scalability and performance** and is built with real-world challenges in mind.

The focus areas for CrypTen and PySyft are naturally aligned and complement each other. The former focuses on building support for various secure and privacy preserving techniques on PyTorch through an encrypted tensor abstraction, while the latter focuses on end user use cases like deployment on edge devices and a user friendly data science platform.

Working together will enable PySyft to use CrypTen as a backend for encrypted tensors. This can lead to an increase in performance for PySyft and the adoption of CrypTen as a runtime by PySyft’s userbase. In addition to this, PyTorch is also adding cryptography friendly features such as support for cryptographically secure random number generation. Over the long run, this allows each library to focus exclusively on its core competencies while enjoying the benefits of the synergistic relationship.

## New Funding for OpenMined Contributors

We are especially excited to announce that the PyTorch team has invested $250,000 to support OpenMined in furthering the development and proliferation of privacy-preserving ML. This gift will be facilitated via the [RAAIS Foundation](https://www.raais.org/) and will be available immediately to support paid fellowship grants for the OpenMined community.

## How to get involved

Thanks to the support from the PyTorch team, OpenMined is able to offer three different opportunities for you to participate in the project’s development. Each of these fellowships furthers our shared mission to lower the barrier-to-entry for privacy-preserving ML and to create a more privacy-preserving world.

### Core PySyft CrypTen Integration Fellowships

During these fellowships, we will integrate CrypTen as a supported backend for encrypted computation in PySyft. This will allow for the high-performance, secure multi-party computation capabilities of CrypTen to be used alongside other important tools in PySyft such as differential privacy and federated learning. For more information on the roadmap and how to apply for a paid fellowship, check out the project’s [call for contributors](https://blog.openmined.org/openmined-pytorch-fellowship-crypten-project).

### Federated Learning on Mobile, Web, and IoT Devices

During these fellowships, we will be extending PyTorch with the ability to perform federated learning across mobile, web, and IoT devices. To this end, a PyTorch front-end will be able to coordinate across federated learning backends that run in Javascript, Kotlin, Swift, and Python. Furthermore, we will also extend PySyft with the ability to coordinate these backends using peer-to-peer connections, providing low latency and the ability to run secure aggregation as a part of the protocol. For more information on the roadmap and how to apply for a paid fellowship, check out the project’s [call for contributors](https://blog.openmined.org/announcing-the-pytorch-openmined-federated-learning-fellowships).

### Development Challenges

Over the coming months, we will issue regular open competitions for increasing the performance and security of the PySyft and PyGrid codebases. For performance-related challenges, contestants will compete (for a cash prize) to make a specific PySyft demo (such as federated learning) as fast as possible. For security-related challenges, contestants will compete to hack into a PyGrid server. The first to demonstrate their ability will win the cash bounty! For more information on the challenges and to sign up to receive emails when each challenge is opened, [sign up here](http://blog.openmined.org/announcing-the-openmined-pytorch-development-challenges).

To apply, select one of the above projects and identify a role that matches your strengths!

Cheers,

Andrew, Laurens, Joe, and Shubho
