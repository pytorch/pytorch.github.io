---
layout: blog_detail
title: "PyTorch at GTC 2025"
author: "Team PyTorch at NVIDIA"
---

[GTC](https://www.nvidia.com/gtc/) is coming back to San Jose on March 17–21, 2025. Join PyTorch Foundation members Arm, AWS, Google Cloud, IBM, Lightning AI, Meta, Microsoft Azure, Snowake, and thousands of developers as we celebrate PyTorch. Together learn how AI & accelerated computing are helping humanity solve our most complex challenges.

Join in person with [discounted GTC registration](https://www.nvidia.com/gtc/?ncid=GTC-NVI0K8HVX) for PyTorch Foundation or [watch online](https://register.nvidia.com/flow/nvidia/gtcs25/registration/) with free registration.


![book cover](/assets/images/pytorch-at-gtc.jpg){:style="width:100%"}


### [Scaling Open Source AI: From Foundation Models to Ecosystem Success](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=pytorch#/session/1738966749087001K1dG)

Hear from PyTorch Foundation’s Executive Director Matt White & panelists from UC Berkeley, Meta, NVIDIA, & Sequoia Capital how open source is transforming AI development, bringing together experts from industry, academia, and venture capital to discuss the technical and business aspects of collaborative open source AI development They’ll examine how open source projects like PyTorch, vLLM, Ray, and NVIDIA's NeMo are accelerating AI innovation while creating new opportunities for businesses and researchers. They'll share real-world experiences from PyTorch's development, Berkeley's research initiatives, and successful AI startups. Take away valuable insights into the technical and business aspects of open source AI. – Monday, Mar 17 10:00 AM - 11:00 AM PDT		


## PyTorch @ GTC

[The Performance of CUDA with the Flexibility of PyTorch ](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=pytorch#/session/1726155993061001WWZM)

Mark Saroufim, Software Engineer, Meta Platforms

This talk explores how PyTorch users are also becoming CUDA developers. We'll start with motivating examples from eager, the launch of torch.compile and the more recent trend of kernel zoos. We will share details on how we went about integrating low bit matmuls in torchao and the torch.compile CUTLASS backend. We'll also discuss details on how you can define, build and package your own custom ops in PyTorch so you get the raw performance of CUDA while maintaining the flexibility of PyTorch.

[Make My PyTorch Model Fast, and Show Me How You Did It](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=pytorch#/session/1727978036338001UVLu)

Thomas Viehmann, Principal Research Engineer, Lightning AI

Luca Antiga, CTO, Lightning AI

PyTorch is popular in deep learning and LLMs for richness and ease of expressions. To make the most of compute resources, PyTorch models benefit from nontrivial optimizations, but this means losing some of their ease and understandability. Learn how with Thunder, a PyTorch-to-Python compiler focused on usability, understandability, and extensibility, you can optimize and transform (i.e., distribute across many machines) models while • leaving the PyTorch code unchanged • targeting a variety of models without needing to adapt to each of them • understanding each transformation step because the results are presented as simple Python code • accessing powerful extension code for your own optimizations with just one or a few lines of code We'll show how the combination of Thunder transforms and the NVIDIA stack (NVFuser, cuDNN, Apex) delivers optimized performance in training and inference on a variety of models.

[FlexAttention: The Flexibility of PyTorch With the Performance of FlashAttention](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=pytorch#/session/1726184633014001Jh5G)

Driss Guessous, Machine Learning Engineer, Meta Platforms

Introducing FlexAttention: a novel PyTorch API that enables custom, user-defined attention mechanisms with performance comparable to state-of-the-art solutions. By leveraging the PyTorch compiler stack, FlexAttention supports dynamic modifications to attention scores within SDPA, achieving both runtime and memory efficiency through kernel fusion with the FlashAttention algorithm. Our benchmarks on A100 GPUs show FlexAttention achieves 90% of FlashAttention2's performance in forward passes and 85% in backward passes. On H100 GPUs, FlexAttention's forward performance averages 85% of FlashAttention3 and is ~25% faster than FlashAttention2, while backward performance averages 76% of FlashAttention3 and is ~3% faster than FlashAttention2. Explore how FlexAttention balances near-state-of-the-art performance with unparalleled flexibility, empowering researchers to rapidly iterate on attention mechanisms without sacrificing efficiency.

[Keep Your GPUs Going Brrr : Crushing Whitespace in Model Training](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=pytorch#/session/1731693095418001cruA)

 Syed Ahmed, Senior Software Engineer, NVIDIA

Alban Desmaison, Research Engineer, Meta

Aidyn Aitzhan, Senior Software Engineer, NVIDIA

Substantial progress has recently been made on the compute-intensive portions of model training, such as high-performing attention variants. While invaluable, this progress exposes previously hidden bottlenecks in model training, such as redundant copies during collectives and data loading time. We'll present recent improvements in PyTorch achieved through Meta/NVIDIA collaboration to tackle these newly exposed bottlenecks and how practitioners can leverage them.

[Accelerated Python: The Community and Ecosystem](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=pytorch#/session/1727176757800001qp7T)

Andy Terrel, CUDA Python Product Lead, NVIDIA

Jeremy Tanner, Open Source Programs, NVIDIA

Anshuman Bhat, CUDA Product Management, NVIDIA

Python is everywhere. Simulation, data science, and Gen AI all depend on it. Unfortunately, the dizzying array of tools leaves a newcomer baffled at where to start. We'll take you on a guided tour of the vibrant community and ecosystem surrounding accelerated Python programming. Explore a variety of tools, libraries, and frameworks that enable efficient computation and performance optimization in Python, including CUDA Python, RAPIDS, Warp, and Legate. We'll also discuss integration points with PyData, PyTorch, and JAX communities. Learn about collaborative efforts within the community, including open source projects and contributions that drive innovation in accelerated computing. We'll discuss best practices for leveraging these frameworks to enhance productivity in developing AI-driven applications and conducting large-scale data analyses.

[Supercharge large scale AI with Google Cloud AI hypercomputer (Presented by Google Cloud)](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=pytorch#/session/1734571562315001xMKM)

Deepak Patil, Product Manager, Google Cloud

Rajesh Anantharaman, Product Management Lead, ML Software, Google Cloud

Unlock the potential of your large-scale AI workloads with Google Cloud AI Hypercomputer – a supercomputing architecture designed for maximum performance and efficiency. In this session, we will deep dive into PyTorch and JAX stacks on Google Cloud on NVIDIA GPUs, and showcase capabilities for high performance foundation model building on Google Cloud.

[Peering Into the Future: What AI and Graph Networks Can Mean for the Future of Financial Analysis](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=pytorch#/session/1739906058885001OxEF)

Siddharth Samsi, Sr. Solutions Architect, NVIDIA

Sudeep Kesh, Chief Innovation Officer, S&P Global

Artificial Intelligence, agentic systems, and graph neural networks (GNNs) are providing the new frontier to assess, monitor, and estimate opportunities and risks across work portfolios within financial services. Although many of these technologies are still developing, organizations are eager to understand their potential. See how S&P Global and NVIDIA are working together to find practical ways to learn and integrate such capabilities, ranging from forecasting corporate debt issuance to understanding capital markets at a deeper level. We'll show a graph representation of market data using the PyTorch-Geometric library and a dataset of issuances spanning three decades and across financial and non-financial industries. Technical developments include generation of a bipartite graph and link-prediction GNN forecasting. We'll address data preprocessing, pipelines, model training, and how these technologies can broaden capabilities in an increasingly complex world.

[Unlock Deep Learning Performance on Blackwell With cuDNN](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=pytorch#/session/1727984645671001Y9eq)

Yang Xu (Enterprise Products), DL Software Engineering Manager, NVIDIA

Since its launch, cuDNN, a library for GPU-accelerating deep learning (DL) primitives, has been powering many AI applications in domains such as conversational AI, recommender systems, and speech recognition, among others. CuDNN remains a core library for DL primitives in popular frameworks such as PyTorch, JAX, Tensorflow, and many more while covering training, fine-tuning, and inference use cases. Even in the rapidly evolving space of Gen AI — be it Llama, Gemma, or mixture-of-experts variants requiring complex DL primitives such as flash attention variants — cuDNN is powering them all. Learn about new/updated APIs of cuDNN pertaining to Blackwell’s microscaling format, and how to program against those APIs. We'll deep dive into leveraging its graph APIs to build some fusion patterns, such as matmul fusion patterns and fused flash attention from state-of-the-art models. Understand how new CUDA graph support in cuDNN, not to be mistaken with the cuDNN graph API, could be exploited to avoid rebuilding CUDA graphs, offering an alternative to CUDA graph capture with real-world framework usage.

[Train and Serve AI Systems Fast With the Lightning AI Open-Source Stack (Presented by Lightning AI)](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=pytorch#/session/1736347047099001au7y)

Luca Antiga, CTO, Lightning AI

See how the Lightning stack can cover the full life cycle, from data preparation to deployment, with practical examples and particular focus on distributed training and high-performance inference. We'll show examples that focus on new features like support for multi-dimensional parallelism through DTensors, as well as quantization through torchao.


## Connect With Experts (Interactive Sessions)

[Meet the Experts From Deep Learning Framework Teams ](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=pytorch#/session/1728516848639001tO7H) 

Eddie Yan, Technical Lead of PyTorch, NVIDIA

Masaki Kozuki, Senior Software Engineer in PyTorch, NVIDIA

Patrick Wang (Enterprise Products), Software Engineer in PyTorch, NVIDIA

Mike Ruberry, Distinguished Engineer in Deep Learning Frameworks, NVIDIA

Rishi Puri, Sr. Deep Learning Engineer and Lead for PyTorch Geometric, NVIDIA


## Training Labs

[Kernel Optimization for AI and Beyond: Unlocking the Power of Nsight Compute ](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=pytorch#/session/1726073884811001C0za)

 Felix Schmitt, Sr. System Software Engineer, NVIDIA

Peter Labus, Senior System Software Engineer, NVIDIA

Learn how to unlock the full potential of NVIDIA GPUs with the powerful profiling and analysis capabilities of Nsight Compute. AI workloads are rapidly increasing the demand for GPU computing, and ensuring that they efficiently utilize all available GPU resources is essential. Nsight Compute is the most powerful tool for understanding kernel execution behavior and performance. Learn how to configure and launch profiles customized for your needs, including advice on profiling accelerated Python applications, AI frameworks like PyTorch, and optimizing Tensor Core utilization essential to modern AI performance. Learn how to debug your kernel and use the expert system built into Nsight Compute, known as “Guided Analysis,” that automatically detects common issues and directs you to the most relevant performance data all the way down to the source code level.

[Make Retrieval Better: Fine-Tuning an Embedding Model for Domain-Specific RAG](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=pytorch#/session/1725042189130001cmoW) 

Gabriel Moreira, Sr. Research Scientist, NVIDIA

Ronay Ak, Sr. Data Scientist, NVIDIA

LLMs power AI applications like conversational chatbots and content generators, but are constrained by their training data. This might lead to hallucinations in content generation, which requires up-to-date or domain-specific information. Retrieval augmented generation (RAG) addresses this issue by enabling LLMs to access external context without modifying model parameters. Embedding or dense retrieval models are a key component of a RAG pipeline for retrieving relevant context to the LLM. However, an embedding model’s effectiveness to capture the unique characteristics of the custom data hinges on the quality and domain relevance of its training data. Fine-tuning embedding models is gaining interest to provide more accurate and relevant responses tailored to users’ specific domain.

In this lab, you'll learn to generate a synthetic dataset with question-context pairs from a domain-specific corpus, and process the data for fine-tuning. Then, fine-tune a text embedding model using synthetic data and evaluate it.


## Poster Presentations

[Single-View X-Ray 3D Reconstruction Using Neural Back Projection and Frustum Resampling](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=pytorch#/session/1729781473379001KiPD)

Tran Minh Quan, Developer Technologist, NVIDIA

[Enable Novel Applications in the New AI Area in Medicine: Accelerated Feature Computation for Pathology Slides](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=pytorch#/session/1729757102989001KDG4)

Nils Bruenggel, Principal Software Engineer, Roche Diagnostics Int. AG