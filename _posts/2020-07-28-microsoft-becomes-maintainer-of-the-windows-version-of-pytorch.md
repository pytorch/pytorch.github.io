---
layout: blog_detail
title: 'Microsoft becomes maintainer of the Windows version of PyTorch'
author: Maxim Lukiyanov - Principal PM at Microsoft, Emad Barsoum - Group EM at Microsoft, Guoliang Hua - Principal EM at Microsoft, Nikita Shulga - Tech Lead at Facebook, Geeta Chauhan - PE Lead at Facebook, Chris Gottbrath - Technical PM at Facebook, Jiachen Pu - Engineer at Facebook

---

Along with the PyTorch 1.6 release, we are excited to announce that Microsoft has expanded its participation in the PyTorch community and will be responsible for the development and maintenance of the PyTorch build for Windows.

According to the latest [Stack Overflow developer survey](https://insights.stackoverflow.com/survey/2020#technology-developers-primary-operating-systems), Windows remains the primary operating system for the developer community (46% Windows vs 28% MacOS). [Jiachen Pu](https://github.com/peterjc123) initially made a heroic effort to add support for PyTorch on Windows, but due to limited resources, Windows support for PyTorch has lagged behind other platforms. Lack of test coverage resulted in unexpected issues popping up every now and then. Some of the core tutorials, meant for new users to learn and adopt PyTorch, would fail to run. The installation experience was also not as smooth, with the lack of official PyPI support for PyTorch on Windows. Lastly, some of the PyTorch functionality was simply not available on the Windows platform, such as the TorchAudio domain library and distributed training support. To help alleviate this pain, Microsoft is happy to bring its Windows expertise to the table and bring PyTorch on Windows to its best possible self.

In the PyTorch 1.6 release, we have improved the core quality of the Windows build by bringing test coverage up to par with Linux for core PyTorch and its domain libraries and by automating tutorial testing. Thanks to the broader PyTorch community, which contributed TorchAudio support to Windows, we were able to add test coverage to all three domain libraries: TorchVision, TorchText and TorchAudio. In subsequent releases of PyTorch, we will continue improving the Windows experience based on community feedback and requests. So far, the feedback we received from the community points to distributed training support and a better installation experience using pip as the next areas of improvement. 

In addition to the native Windows experience, Microsoft released a preview adding [GPU compute support to Windows Subsystem for Linux (WSL) 2](https://blogs.windows.com/windowsdeveloper/2020/06/17/gpu-accelerated-ml-training-inside-the-windows-subsystem-for-linux/) distros, with a focus on enabling AI and ML developer workflows. WSL is designed for developers that want to run any Linux based tools directly on Windows. This preview enables valuable scenarios for a variety of frameworks and Python packages that utilize [NVIDIA CUDA](https://developer.nvidia.com/cuda/wsl) for acceleration and only support Linux. This means WSL customers using the preview can run native Linux based PyTorch applications on Windows unmodified without the need for a traditional virtual machine or a dual boot setup.

## Getting started with PyTorch on Windows
It's easy to get started with PyTorch on Windows. To install PyTorch using Anaconda with the latest GPU support, run the command below. To install different supported configurations of PyTorch, refer to the installation instructions on [pytorch.org](https://pytorch.org).

`conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`

Once you install PyTorch, learn more by visiting the [PyTorch Tutorials](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) and [documentation](https://pytorch.org/docs/stable/index.html).

<div class="text-center">
  <img src="{{ site.url }}/assets/images/pytorch1.6.png" width="100%">
</div>

## Getting started with PyTorch on Windows Subsystem for Linux
The [preview of NVIDIA CUDA support in WSL](https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl) is now available to Windows Insiders running Build 20150 or higher. In WSL, the command to install PyTorch using Anaconda is the same as the above command for native Windows. If you prefer pip, use the command below.

`pip install torch torchvision`

You can use the same tutorials and documentation inside your WSL environment as on native Windows. This functionality is still in preview so if you run into issues with WSL please share feedback via the [WSL GitHub repo](https://github.com/microsoft/WSL) or with NVIDIA CUDA support share via NVIDIA’s [Community Forum for CUDA on WSL](https://forums.developer.nvidia.com/c/accelerated-computing/cuda/cuda-on-windows-subsystem-for-linux/303).

## Feedback
If you find gaps in the PyTorch experience on Windows, please let us know on the [PyTorch discussion forum](https://discuss.pytorch.org/c/windows/26) or file an issue on [GitHub](https://github.com/pytorch/pytorch) using the #module: windows label.
