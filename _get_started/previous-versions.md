---
layout: get_started
title: Previous PyTorch Versions
permalink: /get-started/previous-versions/
background-class: get-started-background
body-class: get-started
order: 4
published: true
redirect_from: /previous-versions.html
---

## Installing previous versions of PyTorch

We'd prefer you install the [latest version](https://pytorch.org/get-started/locally),
but old binaries and installation instructions are provided below for
your convenience.

### Via conda

> This should be used for most previous macOS version installs.

To install a previous version of PyTorch via Anaconda or Miniconda,
replace "0.4.1" in the following commands with the desired version
(i.e., "0.2.0").

Installing with CUDA 9

`conda install pytorch=0.4.1 cuda90 -c pytorch`

or

`conda install pytorch=0.4.1 cuda92 -c pytorch`

Installing with CUDA 8

`conda install pytorch=0.4.1 cuda80 -c pytorch`

Installing with CUDA 7.5

`conda install pytorch=0.4.1 cuda75 -c pytorch`

Installing without CUDA

`conda install pytorch=0.4.1 -c pytorch`

### From source

It is possible to checkout an older version of [PyTorch](https://github.com/pytorch/pytorch)
and build it.
You can list tags in PyTorch git repository with `git tag` and checkout a
particular one (replace '0.1.9' with the desired version) with

`git checkout v0.1.9`

Follow the install from source instructions in the README.md of the PyTorch
checkout.

### Via pip

Download the `whl` file with the desired version via this command (you can replace 1.0.1 with the version you choose):

```
pip install torch==1.0.1 -f https://download.pytorch.org/whl/cpu/stable # CPU-only build
pip install torch==1.0.1 -f https://download.pytorch.org/whl/cu80/stable # CUDA 8.0 build
pip install torch==1.0.1 -f https://download.pytorch.org/whl/cu90/stable # CUDA 9.0 build
pip install torch==1.0.1 -f https://download.pytorch.org/whl/cu92/stable # CUDA 9.2 build
pip install torch==1.0.1 -f https://download.pytorch.org/whl/cu100/stable # CUDA 10.0 build
```

Note: most pytorch versions are available only for specific CUDA versions. For example pytorch=1.0.1 is not available for CUDA 9.2

### (Old) PyTorch Linux binaries compiled with CUDA 7.5

These predate the mechanism above with `pip install -f` and have to be manually installed by downloading the wheel file and `pip install downloaded_file`

- [cu75/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl)
- [cu75/torch-0.3.0.post4-cp27-cp27m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.3.0.post4-cp27-cp27m-linux_x86_64.whl)
- [cu75/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post3-cp27-cp27m-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp27-cp27m-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post2-cp36-cp36m-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post2-cp36-cp36m-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post2-cp35-cp35m-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post2-cp35-cp35m-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post2-cp27-cp27mu-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post2-cp27-cp27mu-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post2-cp27-cp27m-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post2-cp27-cp27m-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post1-cp36-cp36m-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post1-cp36-cp36m-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post1-cp35-cp35m-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post1-cp35-cp35m-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post1-cp27-cp27mu-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post1-cp27-cp27mu-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post1-cp27-cp27m-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post1-cp27-cp27m-manylinux1_x86_64.whl)
- [cu75/torch-0.1.12.post2-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.12.post2-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.12.post1-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.12.post1-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.1.12.post1-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.12.post1-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.12.post1-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.12.post1-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.11.post5-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.11.post5-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.1.11.post5-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.11.post5-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.11.post5-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.11.post5-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.11.post4-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.11.post4-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.1.11.post4-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.11.post4-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.11.post4-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.11.post4-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.10.post2-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.10.post2-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.1.10.post2-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.10.post2-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.10.post2-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.10.post2-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.10.post1-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.10.post1-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.1.10.post1-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.10.post1-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.10.post1-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.10.post1-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.9.post2-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.9.post2-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.1.9.post2-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.9.post2-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.9.post2-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.9.post2-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.9.post1-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.9.post1-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.1.9.post1-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.9.post1-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.9.post1-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.9.post1-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.8.post1-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.8.post1-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.1.8.post1-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.8.post1-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.8.post1-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.8.post1-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.7.post2-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.7.post2-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.1.7.post2-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.7.post2-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.7.post2-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.7.post2-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.6.post22-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.6.post22-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.6.post22-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.6.post22-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.6.post20-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.6.post20-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.6.post20-cp27-cp27mu-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.6.post20-cp27-cp27mu-linux_x86_64.whl)

### Windows binaries

- [cpu/torch-1.0.0-cp35-cp35m-win_amd64.whl](https://download.pytorch.org/whl/cpu/torch-1.0.0-cp35-cp35m-win_amd64.whl)
- [cu80/torch-1.0.0-cp35-cp35m-win_amd64.whl](https://download.pytorch.org/whl/cu80/torch-1.0.0-cp35-cp35m-win_amd64.whl)
- [cu90/torch-1.0.0-cp35-cp35m-win_amd64.whl](https://download.pytorch.org/whl/cu90/torch-1.0.0-cp35-cp35m-win_amd64.whl)
- [cu100/torch-1.0.0-cp35-cp35m-win_amd64.whl](https://download.pytorch.org/whl/cu100/torch-1.0.0-cp35-cp35m-win_amd64.whl)
- [cpu/torch-1.0.0-cp36-cp36m-win_amd64.whl](https://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-win_amd64.whl)
- [cu80/torch-1.0.0-cp36-cp36m-win_amd64.whl](https://download.pytorch.org/whl/cu80/torch-1.0.0-cp36-cp36m-win_amd64.whl)
- [cu90/torch-1.0.0-cp36-cp36m-win_amd64.whl](https://download.pytorch.org/whl/cu90/torch-1.0.0-cp36-cp36m-win_amd64.whl)
- [cu100/torch-1.0.0-cp36-cp36m-win_amd64.whl](https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-win_amd64.whl)
- [cpu/torch-1.0.0-cp37-cp37m-win_amd64.whl](https://download.pytorch.org/whl/cpu/torch-1.0.0-cp37-cp37m-win_amd64.whl)
- [cu80/torch-1.0.0-cp37-cp37m-win_amd64.whl](https://download.pytorch.org/whl/cu80/torch-1.0.0-cp37-cp37m-win_amd64.whl)
- [cu90/torch-1.0.0-cp37-cp37m-win_amd64.whl](https://download.pytorch.org/whl/cu90/torch-1.0.0-cp37-cp37m-win_amd64.whl)
- [cu100/torch-1.0.0-cp37-cp37m-win_amd64.whl](https://download.pytorch.org/whl/cu100/torch-1.0.0-cp37-cp37m-win_amd64.whl)
- [cpu/torch-0.4.1-cp35-cp35m-win_amd64.whl](https://download.pytorch.org/whl/cpu/torch-0.4.1-cp35-cp35m-win_amd64.whl)
- [cu80/torch-0.4.1-cp35-cp35m-win_amd64.whl](https://download.pytorch.org/whl/cu80/torch-0.4.1-cp35-cp35m-win_amd64.whl)
- [cu90/torch-0.4.1-cp35-cp35m-win_amd64.whl](https://download.pytorch.org/whl/cu90/torch-0.4.1-cp35-cp35m-win_amd64.whl)
- [cu92/torch-0.4.1-cp35-cp35m-win_amd64.whl](https://download.pytorch.org/whl/cu92/torch-0.4.1-cp35-cp35m-win_amd64.whl)
- [cpu/torch-0.4.1-cp36-cp36m-win_amd64.whl](https://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-win_amd64.whl)
- [cu80/torch-0.4.1-cp36-cp36m-win_amd64.whl](https://download.pytorch.org/whl/cu80/torch-0.4.1-cp36-cp36m-win_amd64.whl)
- [cu90/torch-0.4.1-cp36-cp36m-win_amd64.whl](https://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-win_amd64.whl)
- [cu92/torch-0.4.1-cp36-cp36m-win_amd64.whl](https://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-win_amd64.whl)
- [cpu/torch-0.4.1-cp37-cp37m-win_amd64.whl](https://download.pytorch.org/whl/cpu/torch-0.4.1-cp37-cp37m-win_amd64.whl)
- [cu80/torch-0.4.1-cp37-cp37m-win_amd64.whl](https://download.pytorch.org/whl/cu80/torch-0.4.1-cp37-cp37m-win_amd64.whl)
- [cu90/torch-0.4.1-cp37-cp37m-win_amd64.whl](https://download.pytorch.org/whl/cu90/torch-0.4.1-cp37-cp37m-win_amd64.whl)
- [cu92/torch-0.4.1-cp37-cp37m-win_amd64.whl](https://download.pytorch.org/whl/cu92/torch-0.4.1-cp37-cp37m-win_amd64.whl)

### Mac and misc. binaries

For recent macOS binaries, use `conda`:

e.g.,

`conda install pytorch=0.4.1 cuda90 -c pytorch`
`conda install pytorch=0.4.1 cuda92 -c pytorch`
`conda install pytorch=0.4.1 cuda80 -c pytorch`
`conda install pytorch=0.4.1 -c pytorch` # No CUDA

- [torchvision-0.1.6-py3-none-any.whl](https://download.pytorch.org/whl/torchvision-0.1.6-py3-none-any.whl)
- [torchvision-0.1.6-py2-none-any.whl](https://download.pytorch.org/whl/torchvision-0.1.6-py2-none-any.whl)
- [torch-1.0.0-cp37-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/cpu/torch-1.0.0-cp37-none-macosx_10_7_x86_64.whl)
- [torch-1.0.0-cp36-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-none-macosx_10_7_x86_64.whl)
- [torch-1.0.0-cp35-none-macosx_10_6_x86_64.whl](https://download.pytorch.org/whl/cpu/torch-1.0.0-cp35-none-macosx_10_6_x86_64.whl)
- [torch-1.0.0-cp27-none-macosx_10_6_x86_64.whl](https://download.pytorch.org/whl/cpu/torch-1.0.0-cp27-none-macosx_10_6_x86_64.whl)
- [torch-0.4.0-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.4.0-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.4.0-cp35-cp35m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.4.0-cp35-cp35m-macosx_10_6_x86_64.whl)
- [torch-0.4.0-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.4.0-cp27-none-macosx_10_6_x86_64.whl)
- [torch-0.3.1-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.3.1-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.3.1-cp35-cp35m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.3.1-cp35-cp35m-macosx_10_6_x86_64.whl)
- [torch-0.3.1-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.3.1-cp27-none-macosx_10_6_x86_64.whl)
- [torch-0.3.0.post4-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.3.0.post4-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.3.0.post4-cp35-cp35m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.3.0.post4-cp35-cp35m-macosx_10_6_x86_64.whl)
- [torch-0.3.0.post4-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.3.0.post4-cp27-none-macosx_10_6_x86_64.whl)
- [torch-0.2.0.post3-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.2.0.post3-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.2.0.post3-cp35-cp35m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.2.0.post3-cp35-cp35m-macosx_10_7_x86_64.whl)
- [torch-0.2.0.post3-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.2.0.post3-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.2.0.post2-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.2.0.post2-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.2.0.post2-cp35-cp35m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.2.0.post2-cp35-cp35m-macosx_10_7_x86_64.whl)
- [torch-0.2.0.post2-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.2.0.post2-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.2.0.post1-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.2.0.post1-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.2.0.post1-cp35-cp35m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.2.0.post1-cp35-cp35m-macosx_10_7_x86_64.whl)
- [torch-0.2.0.post1-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.2.0.post1-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.12.post2-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.12.post2-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.1.12.post2-cp35-cp35m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.12.post2-cp35-cp35m-macosx_10_7_x86_64.whl)
- [torch-0.1.12.post2-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.12.post2-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.12.post1-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.12.post1-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.1.12.post1-cp35-cp35m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.12.post1-cp35-cp35m-macosx_10_7_x86_64.whl)
- [torch-0.1.12.post1-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.12.post1-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.11.post5-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.11.post5-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.1.11.post5-cp35-cp35m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.11.post5-cp35-cp35m-macosx_10_7_x86_64.whl)
- [torch-0.1.11.post5-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.11.post5-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.11.post4-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.11.post4-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.1.11.post4-cp35-cp35m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.11.post4-cp35-cp35m-macosx_10_7_x86_64.whl)
- [torch-0.1.11.post4-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.11.post4-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.10.post1-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.10.post1-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.1.10.post1-cp35-cp35m-macosx_10_6_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.10.post1-cp35-cp35m-macosx_10_6_x86_64.whl)
- [torch-0.1.10.post1-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.10.post1-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.9.post2-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.9.post2-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.1.9.post2-cp35-cp35m-macosx_10_6_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.9.post2-cp35-cp35m-macosx_10_6_x86_64.whl)
- [torch-0.1.9.post2-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.9.post2-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.9.post1-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.9.post1-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.1.9.post1-cp35-cp35m-macosx_10_6_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.9.post1-cp35-cp35m-macosx_10_6_x86_64.whl)
- [torch-0.1.9.post1-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.9.post1-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.8.post1-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.8.post1-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.1.8.post1-cp35-cp35m-macosx_10_6_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.8.post1-cp35-cp35m-macosx_10_6_x86_64.whl)
- [torch-0.1.8.post1-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.8.post1-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.7.post2-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.7.post2-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.1.7.post2-cp35-cp35m-macosx_10_6_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.7.post2-cp35-cp35m-macosx_10_6_x86_64.whl)
- [torch-0.1.7.post2-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.7.post2-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.6.post22-cp35-cp35m-macosx_10_6_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.6.post22-cp35-cp35m-macosx_10_6_x86_64.whl)
- [torch-0.1.6.post22-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.6.post22-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.6.post20-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.6.post20-cp35-cp35m-linux_x86_64.whl)
- [torch-0.1.6.post20-cp27-cp27mu-linux_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.6.post20-cp27-cp27mu-linux_x86_64.whl)
- [torch-0.1.6.post17-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.6.post17-cp35-cp35m-linux_x86_64.whl)
- [torch-0.1.6.post17-cp27-cp27mu-linux_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.6.post17-cp27-cp27mu-linux_x86_64.whl)
- [torch-0.1-cp35-cp35m-macosx_10_6_x86_64.whl](https://download.pytorch.org/whl/torch-0.1-cp35-cp35m-macosx_10_6_x86_64.whl)
- [torch-0.1-cp27-cp27m-macosx_10_6_x86_64.whl](https://download.pytorch.org/whl/torch-0.1-cp27-cp27m-macosx_10_6_x86_64.whl)
- [torch_cuda80-0.1.6.post20-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/torch_cuda80-0.1.6.post20-cp35-cp35m-linux_x86_64.whl)
- [torch_cuda80-0.1.6.post20-cp27-cp27mu-linux_x86_64.whl](https://download.pytorch.org/whl/torch_cuda80-0.1.6.post20-cp27-cp27mu-linux_x86_64.whl)
- [torch_cuda80-0.1.6.post17-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/torch_cuda80-0.1.6.post17-cp35-cp35m-linux_x86_64.whl)
- [torch_cuda80-0.1.6.post17-cp27-cp27mu-linux_x86_64.whl](https://download.pytorch.org/whl/torch_cuda80-0.1.6.post17-cp27-cp27mu-linux_x86_64.whl)

<script type="text/javascript">
  var pageId = "previous-versions"; // TBD: Make this programmatic
  $(".main-content-menu .nav-item").removeClass("nav-select");
  $(".main-content-menu .nav-link[data-id='" + pageId + "']").parent(".nav-item").addClass("nav-select");
</script>
<script src="{{ site.baseurl }}/assets/quick-start-module.js"></script>
<script src="{{ site.baseurl }}/assets/show-screencast.js"></script>
