---
layout: blog_detail
title: "PyTorch Shanghai Meetup Notes"
---

## Summary

![group photo](/assets/images/pytorch-shanghai-notes/fg1.jpg){:style="width:100%"}

We are honored to successfully host the PyTorch Shanghai Meetup on August 15, 2024\. This Meetup has received great attention from the industry. We invited senior PyTorch developers from Intel and Huawei as guest speakers, who shared their valuable experience and the latest technical trends. In addition, this event also attracted PyTorch enthusiasts from many technology companies and well-known universities. A total of more than 40 participants gathered together to discuss and exchange the latest applications and technological advances of PyTorch.  

This Meetup not only strengthened the connection between PyTorch community members, but also provided a platform for local AI technology enthusiasts to learn, communicate and grow. We look forward to the next gathering to continue to promote the development of PyTorch technology in the local area.

## 1\. PyTorch Foundation Updates

![man instructing students](/assets/images/pytorch-shanghai-notes/fg2.jpg){:style="width:100%"}

PyTorch Board member Fred Li shared the latest updates in the PyTorch community, He reviewed the development history of the PyTorch community, explained in detail the growth path of community developers, encouraged everyone to delve deeper into technology, and introduced the upcoming PyTorch Conference 2024 related matters.

## 2\. Intel’s Journey with PyTorch Democratizing AI with ubiquitous hardware and open software

PyTorch CPU module maintainer Jiong Gong shared 6-year technical contributions from Intel to PyTorch and its ecosystem, explored the remarkable advancements that Intel has made in both software and hardware democratizing AI, ensuring accessibility, and optimizing performance across a diverse range of Intel hardware platforms.

![man instructing students](/assets/images/pytorch-shanghai-notes/fg3.jpg){:style="width:100%"}

## 3\. Exploring Multi-Backend Support in PyTorch Ecosystem: A Case Study of Ascend

![man instructing students](/assets/images/pytorch-shanghai-notes/fg4.jpg){:style="width:100%"}

Fengchun Hua, a PyTorch contributor from Huawei, took Huawei Ascend NPU as an example to demonstrate the latest achievements in multi-backend support for PyTorch applications. He introduced the hardware features of Huawei Ascend NPU and the infrastructure of CANN (Compute Architecture for Neural Networks), and explained the key achievements and innovations in native support work. He also shared the current challenges and the next work plan.  

Yuanhao Ji, another PyTorch contributor from Huawei, then introduced the Autoload Device Extension proposal, explained its implementation details and value in improving the scalability of PyTorch, and introduced the latest work progress of the PyTorch Chinese community.

## 4\. Intel XPU Backend for Inductor

![man instructing students](/assets/images/pytorch-shanghai-notes/fg5.jpg){:style="width:100%"}

Eikan is a PyTorch contributor from Intel. He focuses on torch.compile stack for both Intel CPU and GPU. In this session, Eikan presented Intel's efforts on torch.compile for Intel GPUs. He provided updates on the current status of Intel GPUs within PyTorch, covering both functionality and performance aspects. Additionally, Eikan used Intel GPU as a case study to demonstrate how to integrate a new backend into the Inductor using Triton. 

## 5\. PyTorch PrivateUse1 Evolution Approaches and Insights

![man instructing students](/assets/images/pytorch-shanghai-notes/fg6.jpg){:style="width:100%"}

Jiawei Li, a PyTorch collaborator from Huawei, introduced PyTorch's Dispatch mechanism and emphasized the limitations of DIspatchKey. He took Huawei Ascend NPU as an example to share the best practices of the PyTorch PrivateUse1 mechanism. He mentioned that while using the PrivateUse1 mechanism, Huawei also submitted many improvements and bug fixes for the mechanism to the PyTorch community. He also mentioned that due to the lack of upstream CI support for out-of-tree devices, changes in upstream code may affect their stability and quality, and this insight was recognized by everyone.  
