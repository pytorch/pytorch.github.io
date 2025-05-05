---
layout: blog_detail
title: 'Recap of the PyTorch Korea User Group Meetup: A Technical Conference with a PyTorch Core Maintainer'
author: 'Jiho Kim, PyTorch Korea User Group'
---

At the end of March, the PyTorch Korea User Group hosted a special meetup that brought together prominent speakers for deep discussions on the PyTorch core and its broader ecosystem. With the event more than doubling in size compared to past gatherings, we were able to connect with even more developers and share insights. Huge thanks to [goorm](https://goorm.co/) for sponsoring the fantastic venue! 😄


![people at a conference](/assets/images/pt-korea-user-group-recap/fg1.jpg){:style="width:100%"}



This recap is for those who couldn’t attend in person, as well as for participants who want to revisit the energy and insights of the day. The event featured experts in core PyTorch, AI accelerators, inference optimization, and large language model development. Below is a quick overview of the key sessions that anchored the conference.



## 1️⃣ Jerry Lee | PyTorch Foundation

Representing the PyTorch Foundation, part of the Linux Foundation, Jaeung provided an overview of how PyTorch is driving core open source technologies forward. He shared PyTorch's growth story, the many global projects currently in motion, and the ecosystem’s impressive 20%+ annual growth. The session also covered how the foundation operates, how member organizations are involved, and upcoming plans that are particularly useful for practitioners.


![people at a conference](/assets/images/pt-korea-user-group-recap/fg2.jpg){:style="width:100%"}


## 2️⃣ Alban Desmaison | PyTorch Roadmap

Alban shared the design philosophy behind PyTorch and Meta’s official contribution roadmap ([link](https://dev-discuss.pytorch.org/t/meta-pytorch-team-2025-h1-roadmaps/2794)). He provided a deep technical dive into the differences between Eager and Compiled modes, especially breaking down the backend architecture of device Eager execution. Practical tools and improvements were also introduced—such as memory profilers, enhanced custom operator support, and pinned memory optimizations.


![people at a conference](/assets/images/pt-korea-user-group-recap/fg3.jpg){:style="width:100%"}




## 3️⃣ Hongseok Kim | PyTorch on Rebellions AI Accelerators: Status

Rebellions is building runtime integration for their proprietary NPU architecture, fully aligned with the structural changes in PyTorch 2.0. This talk introduced the performance and scalability of their upcoming chip, their integration strategy with the PyTorch runtime, and challenges in supporting Eager Mode. Hongseok also previewed their roadmap toward releasing these features within the year.

![people at a conference](/assets/images/pt-korea-user-group-recap/fg4.jpg){:style="width:100%"}



## 4️⃣ Kyujin Cho | Backend.AI: A Unified Platform for All AI Accelerators

Backend.AI abstracts and integrates various AI accelerators into a unified workflow. As the diversity of accelerator architectures grows, the need for portability and infrastructure unification becomes even more important. This session showcased features across development and operations—from NPU scheduling and resource allocation to monitoring. Backend.AI currently supports accelerators from NVIDIA, Intel, Tenstorrent, Rebellions, and more.

![people at a conference](/assets/images/pt-korea-user-group-recap/fg5.jpg){:style="width:100%"}



## 5️⃣ Taeho Kim | Optimizing & Deploying Models Across Multiple Chipsets Using NetsPresso

This talk focused on the challenges of inference in real-world industrial applications of AI models. As new state-of-the-art models emerge rapidly, there’s a growing need for environments that can quickly validate device compatibility—ideally with one-click ease. NetsPresso is actively working on a static graph representation compatible with PyTorch, offering efficient support for model development, optimization, and testing.


![people at a conference](/assets/images/pt-korea-user-group-recap/fg6.jpg){:style="width:100%"}


## 6️⃣ Jungyeop Lee | The Journey to Reproduce Deepseek-R1

Jungyeop took us through his journey of reproducing Deepseek, a large language model—an effort that involved 201 experiments. He shared real-world lessons from training with Korean data, tokenizer modifications, and fine-tuning strategies. His practical insights and next steps were especially valuable for those building or re-implementing large models from scratch.


![people at a conference](/assets/images/pt-korea-user-group-recap/fg7.jpg){:style="width:100%"}


## 7️⃣ Sol Kim | A journey from TCP architecture to production-level LLMs

Sol presented an integrated optimization approach to deploying large models using the TCP(Tensor Contraction Processor) architecture, which supports tensor contraction at the hardware level. The talk highlighted optimization techniques built on hardware abstraction layers (HALs) and bottom-up integration strategies with PyTorch—offering a hybrid hardware-software perspective.


![people at a conference](/assets/images/pt-korea-user-group-recap/fg8.jpg){:style="width:100%"}

## 💡 Panel Talk & Q&A 💡

The event wrapped up with an engaging panel discussion. Attendees asked sharp questions, and the speakers offered insightful answers. It was a powerful moment that captured the community’s enthusiasm for PyTorch and their hunger for deeper technical understanding.


![people at a conference](/assets/images/pt-korea-user-group-recap/fg9.jpg){:style="width:100%"}


## Final Thoughts

Since our first offline meetup in October 2022, the PyTorch Korea User Group has held five major technical conferences. Each event deepens our appreciation for the scale and depth of the PyTorch ecosystem. With perspectives from users, contributors, and ecosystem builders, the stories we share are only growing—and we’re committed to continuing this journey together.

See you at the next conference—with even more exciting talks to come! 🙌