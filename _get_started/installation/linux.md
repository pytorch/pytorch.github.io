# Installing on Linux
{:.no_toc}

PyTorch can be installed and used on various Linux distributions. Depending on your system and compute requirements, your experience with PyTorch on Linux may vary in terms of processing time. It is recommended, but not required, that your Linux system has an NVIDIA GPU in order to harness the full power of PyTorch's [CUDA](https://developer.nvidia.com/cuda-zone) [support](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html?highlight=cuda#cuda-tensors)..

## Prerequisites
{: #linux-prerequisites}

### Supported Linux Distributions

PyTorch is supported on Linux distributions that use [glibc](https://www.gnu.org/software/libc/) >= v2.17, which include the following:

* [Arch Linux](https://www.archlinux.org/download/), minimum version 2012-07-15
* [CentOS](https://www.centos.org/download/), minimum version 7.3-1611
* [Debian](https://www.debian.org/distrib/), minimum version 8.0
* [Fedora](https://getfedora.org/), minimum version 24
* [Mint](https://linuxmint.com/download.php), minimum version 14
* [OpenSUSE](https://software.opensuse.org/), minimum version 42.1
* [PCLinuxOS](https://www.pclinuxos.com/get-pclinuxos/), minimum version 2014.7
* [Slackware](http://www.slackware.com/getslack/), minimum version 14.2
* [Ubuntu](https://www.ubuntu.com/download/desktop), minimum version 13.04

> The install instructions here will generally apply to all supported Linux distributions. An example difference is that your distribution may support `yum` instead of `apt`. The specific examples shown were run on an Ubuntu 18.04 machine.

### Python

Python 3.6 or greater is generally installed by default on any of our supported Linux distributions, which meets our recommendation.

> Tip: By default, you will have to use the command `python3` to run Python. If you want to use just the command `python`, instead of `python3`, you can symlink `python` to the `python3` binary.

However, if you want to install another version, there are multiple ways:

* APT
* [Python website](https://www.python.org/downloads/mac-osx/)

If you decide to use APT, you can run the following command to install it:

```bash
sudo apt install python
```

> PyTorch can be installed with Python 2.7, but it is recommended that you use Python 3.6 or greater, which can be installed via any of the mechanisms above .

> If you use [Anaconda](#anaconda) to install PyTorch, it will install a sandboxed version of Python that will be used for running PyTorch applications.

### Package Manager

To install the PyTorch binaries, you will need to use one of two supported package managers: [Anaconda](https://www.anaconda.com/download/#linux) or [pip](https://pypi.org/project/pip/). Anaconda is the recommended package manager as it will provide you all of the PyTorch dependencies in one, sandboxed install, including Python.

#### Anaconda

To install Anaconda, you will use the [command-line installer](https://www.anaconda.com/download/#linux). Right-click on the 64-bit installer link, select `Copy Link Location`, and then use the following commands:

```bash
# The version of Anaconda may be different depending on when you are installing`
curl -O https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
sh Anaconda3-5.2.0-Linux-x86_64.sh
# and follow the prompts. The defaults are generally good.`
```

> You may have to open a new terminal or re-source your `~/.bashrc `to get access to the `conda` command.

<div>
  <a href="javascript:void(0);" class="btn btn-lg btn-orange btn-demo show-screencast">Show Demo</a>
  <div class="screencast">
    <script src="https://asciinema.org/a/SRbIn2nFnsUiYtlIcUsDVXN4n.js" id="asciicast-SRbIn2nFnsUiYtlIcUsDVXN4n" data-speed="3" async></script>
    <a href="javascript:void(0);" class="btn btn-lg btn-orange btn-demo show-info">Hide Demo</a>
  </div>
</div>

#### pip

*Python 3*

While Python 3.x is installed by default on Linux, `pip` is not installed by default.

```bash
sudo apt install python3-pip
```

> Tip: If you want to use just the command  `pip`, instead of `pip3`, you can symlink `pip` to the `pip3` binary.

*Python 2*

If you are using Python 2.7, you will need to use this command

```bash
sudo apt install python-pip
```

## Installation
{: #linux-installation}

### Anaconda

#### No CUDA

To install PyTorch via Anaconda, and do not have a [CUDA-capable](https://developer.nvidia.com/cuda-zone) system or do not require CUDA, use the following `conda` command.

```bash
conda install pytorch-cpu torchvision-cpu -c pytorch
```

<div>
  <a href="javascript:void(0);" class="btn btn-lg btn-orange btn-demo show-screencast">Show Demo</a>
  <div class="screencast">
    <script src="https://asciinema.org/a/wtojk0bqpUDIb3yIHUsZqD9Ha.js" id="asciicast-wtojk0bqpUDIb3yIHUsZqD9Ha" data-speed="3" async></script>
    <a href="javascript:void(0);" class="btn btn-lg btn-orange btn-demo show-info">Hide Demo</a>
  </div>
</div>

#### CUDA 9.0

To install PyTorch via Anaconda, and you are using CUDA 9.0, use the following `conda` command:

```bash
conda install pytorch torchvision -c pytorch
```

<div>
  <a href="javascript:void(0);" class="btn btn-lg btn-orange btn-demo show-screencast">Show Demo</a>
  <div class="screencast">
    <script src="https://asciinema.org/a/HaCIxmYVEd8xGRAKsDu9hv9up.js" id="asciicast-HaCIxmYVEd8xGRAKsDu9hv9up" data-speed="5" async></script>
    <a href="javascript:void(0);" class="btn btn-lg btn-orange btn-demo show-info ">Hide Demo</a>
  </div>
</div>

#### CUDA 8.x

```bash
conda install pytorch torchvision cuda80 -c pytorch
```

#### CUDA 9.2

```bash
conda install pytorch torchvision cuda92 -c pytorch
```

### pip

#### No CUDA

To install PyTorch via pip, and do not have a [CUDA-capable](https://developer.nvidia.com/cuda-zone) system or do not require CUDA, use the following command, depending on your Python version:

```bash
# Python 2.7
pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl
pip install torchvision

# if the above command does not work, then you have python 2.7 UCS2, use this command
pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp27-cp27m-linux_x86_64.whl
```

```bash
# Python 3.5
pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp35-cp35m-linux_x86_64.whl
pip3 install torchvision
```

```bash
# Python 3.6
pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision
```

```bash
# Python 3.7
pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1.post2-cp37-cp37m-linux_x86_64.whl
pip3 install torchvision
```


#### CUDA 9.0

To install PyTorch via pip, and you are using CUDA 9.0 or do not require CUDA, use the following command, depending on your Python version:

```bash
# Python 3.x
pip3 install torch torchvision
```

```bash
# Python 2.7`
pip install torch torchvision
```

#### CUDA 8.x

```bash
# Python 2.7
pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl
pip install torchvision

# if the above command does not work, then you have python 2.7 UCS2, use this command
pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp27-cp27m-linux_x86_64.whl
```

```bash
# Python 3.5
pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp35-cp35m-linux_x86_64.whl
pip3 install torchvision
```

```bash
# Python 3.6
pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision
```

```bash
# Python 3.7
pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.1.post2-cp37-cp37m-linux_x86_64.whl
pip3 install torchvision
```

#### CUDA 9.2

```bash
# Python 2.7
pip install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl
pip install torchvision

# if the above command does not work, then you have python 2.7 UCS2, use this command
pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp27-cp27m-linux_x86_64.whl
```

```bash
# Python 3.5
pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp35-cp35m-linux_x86_64.whl
pip3 install torchvision
```

```bash
# Python 3.6
pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision
```

```bash
# Python 3.7
pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1.post2-cp37-cp37m-linux_x86_64.whl
pip3 install torchvision
```

## Verification
{: #linux-verification}

To ensure that PyTorch was installed correctly, we can verify the installation by running sample PyTorch code. Here we will construct a randomly initialized tensor.


```python
from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)
```

The output should be something similar to:

```
tensor([[0.3380, 0.3845, 0.3217],
        [0.8337, 0.9050, 0.2650],
        [0.2979, 0.7141, 0.9069],
        [0.1449, 0.1132, 0.1375],
        [0.4675, 0.3947, 0.1426]])
```

Additionally, to check if your GPU driver and CUDA is enabled and accessible by PyTorch, run the following commands to return whether or not the CUDA driver is enabled:

```python
import torch
torch.cuda.is_available()
```

<div>
  <a href="javascript:void(0);" class="btn btn-lg btn-orange btn-demo show-screencast">Show Demo</a>
  <div class="screencast">
    <script src="https://asciinema.org/a/15dyZZvvakqbfKgfh2LByMkXz.js" id="asciicast-15dyZZvvakqbfKgfh2LByMkXz" data-speed="2" async></script>
    <a href="javascript:void(0);" class="btn btn-lg btn-orange btn-demo show-info">Hide Demo</a>
  </div>
</div>

## Building from source
{: #linux-from-source}

For the majority of PyTorch users, installing from a pre-built binary via a package manager will provide the best experience. However, there are times when you may want to install the bleeding edge PyTorch code, whether for testing or actual development on the PyTorch core. To install the latest PyTorch code, you will need to [build PyTorch from source](https://github.com/pytorch/pytorch#from-source).

> You will also need to build from source if you want CUDA support.

### Prerequisites

1. Install Anaconda[#anaconda]
2. Install [CUDA](https://developer.nvidia.com/cuda-downloads), if your machine has a [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus).
3. Install optional dependencies:

```bash
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" # [anaconda root directory]

# Install basic dependencies
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install -c mingfeima mkldnn

# Add LAPACK support for the GPU
conda install -c pytorch magma-cuda80 # or magma-cuda90 if CUDA 9
```

### Build

```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
python setup.py install
```

You can verify the installation as described [above](#linux-verification).
