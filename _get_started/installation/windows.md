# Installing on Windows
{:.no_toc}

PyTorch can be installed and used on various Windows distributions. Depending on your system and compute requirements, your experience with PyTorch on Windows may vary in terms of processing time. It is recommended, but not required, that your Windows system has an NVIDIA GPU in order to harness the full power of PyTorch's [CUDA](https://developer.nvidia.com/cuda-zone) [support](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html?highlight=cuda#cuda-tensors).

## Prerequisites
{: #windows-prerequisites}

### Supported Windows Distributions

PyTorch is supported on the following Windows distributions:

* [Windows](https://www.microsoft.com/en-us/windows) 7 and greater; [Windows 10](https://www.microsoft.com/en-us/software-download/windows10ISO) or greater recommended.
* [Windows Server 2008](https://docs.microsoft.com/en-us/windows-server/windows-server) r2 and greater

> The install instructions here will generally apply to all supported Windows distributions. The specific examples shown will be run on a Windows 10 Enterprise machine

### Python

Currently, PyTorch on Windows only supports Python 3.x; Python 2.x is not supported.

As it is not installed by default on Windows, there are multiple ways to install Python:

* [Chocolatey](https://chocolatey.org/)
* [Python website](https://www.python.org/downloads/windows/)
* [Anaconda](#anaconda)

> If you use Anaconda to install PyTorch, it will install a sandboxed version of Python that will be used for running PyTorch applications.

> If you decide to use Chocolatey, and haven't installed Chocolatey yet, ensure that you are [running your command prompt as an administrator](https://www.howtogeek.com/194041/how-to-open-the-command-prompt-as-administrator-in-windows-8.1/).

For a Chocolatey-based install, run the following command in an [administrative command prompt](https://www.howtogeek.com/194041/how-to-open-the-command-prompt-as-administrator-in-windows-8.1/):

```bash
choco install python
```

### Package Manager

To install the PyTorch binaries, you will need to use at least one of two supported package managers: [Anaconda](https://www.anaconda.com/download/#windows) and [pip](https://pypi.org/project/pip/). Anaconda is the recommended package manager as it will provide you all of the PyTorch dependencies in one, sandboxed install, including Python and `pip.`

#### Anaconda

To install Anaconda, you will use the [64-bit graphical installer](https://www.anaconda.com/download/#windows) for PyTorch 3.x. Click on the installer link and select `Run`. Anaconda will download and the installer prompt will be presented to you. The default options are generally sane.

#### pip

If you installed Python by any of the recommended ways above[LINK], `[pip](https://pypi.org/project/pip/)` will have already been installed for you.

## Installation
{: #windows-installation}

### Anaconda

To install PyTorch with Anaconda, you will need to open an Anaconda prompt via `Start | Anaconda3 | Anaconda Prompt`.

#### No CUDA

To install PyTorch via Anaconda, and do not have a CUDA-capable[LINK] system or do not require CUDA, use the following `conda` command.

```bash
conda install pytorch-cpu -c pytorch
pip3 install torchvision
```

#### CUDA 9.0

To install PyTorch via Anaconda, and you are using CUDA 9.0, use the following `conda` command:

```bash
conda install pytorch -c pytorch
pip3 install torchvision
```

#### CUDA 8.x

```bash
conda install pytorch cuda80 -c pytorch
pip3 install torchvision
```

#### CUDA 9.2

```bash
conda install pytorch cuda92 -c pytorch
pip3 install torchvision
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
pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp35-cp35m-win_amd64.whl
pip3 install torchvision
```

```bash
# Python 3.6
pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-win_amd64.whl
pip3 install torchvision
```

```bash
# Python 3.7
pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp37-cp37m-win_amd64.whl
pip3 install torchvision
```


_CUDA 9.0_

To install PyTorch via pip, and you are using CUDA 9.0, use the following command, depending on your Python version:


```
# Python 3.5
pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp35-cp35m-win_amd64.whl
pip3 install torchvision
```

```
# Python 3.6
pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-win_amd64.whl
pip3 install torchvision
```

```
# Python 3.7
pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp37-cp37m-win_amd64.whl
pip3 install torchvision
```

```
`# Python 3.x`
`pip3 install torch torchvision`
```

```
`# Python 2.7`
`pip install torch torchvision `
```

_CUDA 8.x_

```
# Python 3.5
pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp35-cp35m-win_amd64.whl
pip3 install torchvision
```

```
# Python 3.6
pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp36-cp36m-win_amd64.whl
pip3 install torchvision
```

```
# Python 3.7
pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp37-cp37m-win_amd64.whl
pip3 install torchvision
```

_CUDA 9.2_


```
# Python 3.5
pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp35-cp35m-win_amd64.whl
pip3 install torchvision
```

```
# Python 3.6
pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-win_amd64.whl
pip3 install torchvision
```

```
# Python 3.7
pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp37-cp37m-win_amd64.whl
pip3 install torchvision
```

## Verification
{: #windows-verification}

To ensure that PyTorch was installed correctly, we can verify the installation by running sample PyTorch code. Here we will construct a randomly initialized tensor.

From the command line, type:

```bash
python
```

then enter the following code:

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

## Building from source
{: #windows-from-source}

For the majority of PyTorch users, installing from a pre-built binary via a package manager will provide the best experience. However, there are times when you may want to install the bleeding edge PyTorch code, whether for testing or actual development on the PyTorch core. To install the latest PyTorch code, you will need to [build PyTorch from source](https://github.com/pytorch/pytorch#from-source).

### Prerequisites

1. Install [Anaconda](#anaconda)
2. Install [CUDA](https://developer.nvidia.com/cuda-downloads), if your machine has a [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus).
3. If you want to build on Windows, Visual Studio 2017 14.11 toolset and NVTX are also needed. Especially, for CUDA 8 build on Windows, there will be an additional requirement for VS 2015 Update 3 and a patch for it. The details of the patch can be found out [here](https://support.microsoft.com/en-gb/help/4020481/fix-link-exe-crashes-with-a-fatal-lnk1000-error-when-you-use-wholearch).
4. Install optional dependencies:

```bash
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
```

### Build

```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorchset "VS150COMNTOOLS=C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build"
set CMAKE_GENERATOR=Visual Studio 15 2017 Win64
set DISTUTILS_USE_SDK=1
REM The following two lines are needed for Python 2.7, but the support for it is very experimental.
set MSSdk=1
set FORCE_PY27_BUILD=1
REM As for CUDA 8, VS2015 Update 3 is also required to build PyTorch. Use the following two lines.
set "PREBUILD_COMMAND=%VS140COMNTOOLS%\..\..\VC\vcvarsall.bat"
set PREBUILD_COMMAND_ARGS=x64

call "%VS150COMNTOOLS%\vcvarsall.bat" x64 -vcvars_ver=14.11
python setup.py install
```

You can verify the installation as described [above](#windows-verification).
