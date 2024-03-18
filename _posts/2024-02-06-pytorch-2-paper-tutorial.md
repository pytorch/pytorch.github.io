---
layout: blog_detail
title: "PyTorch 2 paper and tutorial @ ASPLOS 2024"
---

The PyTorch team is excited to share that our paper on PyTorch 2 has been accepted for presentation at the ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), scheduled to take place from April 27 to May 1, 2024, in San Diego, CA, USA. 

The paper delves into the implementation of torch.compile and highlights the key technologies driving it, including TorchDynamo (graph capture), TorchInductor (backend compiler), and Dynamic Shape support.

During the ASPLOS conference, we'll be conducting a tutorial on Saturday, April 27, focusing on the inner workings of PyTorch 2 and how systems researchers can leverage and build upon it. Stay tuned for more details as the event approaches – we look forward to your participation!

A preview of the paper is attached below:

Title: **PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation.** [**Full Paper PDF**](/assets/pytorch2-2.pdf)

### Abstract
This paper introduces two extensions to the popular PyTorch machine learning framework, TorchDynamo and TorchInductor, which implement the torch.compile feature released in PyTorch 2. TorchDynamo is a Python-level just-in-time (JIT) compiler that enables graph compilation in PyTorch programs without sacrificing the flexibility of Python. It achieves this by dynamically modifying Python bytecode before execution and extracting sequences of PyTorch operations into an FX graph, which is then JIT compiled using one of many extensible backends. TorchInductor is the default compiler backend for TorchDynamo, which translates PyTorch programs into OpenAI's Triton for GPUs and C++ for CPUs. Results show that TorchDynamo is able to capture graphs more robustly than prior approaches while adding minimal overhead, and TorchInductor is able to provide a 2.27x inference and 1.41x training geometric mean speedup on an NVIDIA A100 GPU across 180+ real-world models, which outperforms six other compilers. These extensions provide a new way to apply optimizations through compilers in eager mode frameworks like PyTorch.


### Authors

Jason Ansel (Meta); Edward Yang (Meta); Horace He (Meta); Natalia Gimelshein (OpenAI); Animesh Jain (Meta); Michael Voznesensky (Meta); Bin Bao (Meta); Peter Bell (Quansight); David Berard (Meta); Evgeni Burovski Quansight; Geeta Chauhan (Meta); Anjali Chourdia (Meta); Will Constable (Meta); Alban Desmaison (Meta); Zachary DeVito (Meta); Elias Ellison (Meta); Will Feng (Meta); Jiong Gong (Intel); Michael Gschwind (Meta); Brian Hirsh (Meta); Sherlock Huang (Meta); Kshiteej Kalambarkar (Quansight); Laurent Kirsch (Meta); Michael Lazos (Meta); Mario Lezcano (Quansight); Yanbo Liang (Meta); Jason Liang (Meta); Yinghai Lu (Meta); CK Luk (Meta); Bert Maher (Meta); Yunjie Pan (University of Michigan); Christian Puhrsch (Meta); Matthias Reso (Meta); Mark Saroufim (Meta); Marcos Yukio Siraichi (Quansight); Helen Suk (Meta); Michael Suo (Meta); Phil Tillet (OpenAI); Eikan Wang (Intel); Xiaodong Wang (Meta); William Wen (Meta); Shunting Zhang (Meta); Xu Zhao (Meta); Keren Zhou (OpenAI & George Mason University); Richard Zou (Meta); Ajit Mathews (Meta); Gregory Chanan (Meta); Peng Wu (Meta); Soumith Chintala (Meta)

### ASPLOS'24 - Full Day Tutorial Schedule

Full schedule for the ASPLOS'24 PyTorch 2 Tutoral on Saturday, April 27th is available [here](https://github.com/pytorch/workshops/tree/master/ASPLOS_2024)
