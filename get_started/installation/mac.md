# Installing on macOS
{:.no_toc}

PyTorch can be installed and used on macOS. Depending on your system and compute requirements, your experience with PyTorch on a Mac may vary in terms of processing time. It is recommended, but not required, that your Mac have an NVIDIA GPU in order to harness the full power of PyTorch's [CUDA](https://developer.nvidia.com/cuda-zone) [support](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html?highlight=cuda#cuda-tensors).

> Currently, CUDA support on macOS is only available by [building PyTorch from source](#mac-from-source)

## Prerequisites
{: #mac-prerequisites}

### macOS Version

PyTorch is supported on macOS 10.10 (Yosemite) or above.

### Python
{: #mac-python}

It is recommended that you use Python 3.5 or greater, which can be installed either through the Anaconda package manager (see [below](#anaconda)), [Homebrew](https://brew.sh/), or the [Python website](https://www.python.org/downloads/mac-osx/).

### Package Manager
{: #mac-package-manager}

To install the PyTorch binaries, you will need to use one of two supported package managers: [Anaconda](https://www.anaconda.com/download/#macos) or [pip](https://pypi.org/project/pip/). Anaconda is the recommended package manager as it will provide you all of the PyTorch dependencies in one, sandboxed install, including Python.

#### Anaconda

To install Anaconda, you can [download graphical installer](https://www.anaconda.com/download/#macos) or use the command-line installer. If you use the command-line installer, you can right-click on the installer link, select `Copy Link Address`, and then use the following commands:

```bash
# The version of Anaconda may be different depending on when you are installing`
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
sh Miniconda3-latest-MacOSX-x86_64.sh
# and follow the prompts. The defaults are generally good.`
```

#### pip

*Python 3*

If you installed Python via Homebrew or the Python website, `pip` was installed with it. If you installed Python 3.x, then you will be using the command `pip3`.

> Tip: If you want to use just the command  `pip`, instead of `pip3`, you can symlink `pip` to the `pip3` binary.

## Installation
{: #mac-installation}

### Anaconda
{: #mac-anaconda}

To install PyTorch via Anaconda, use the following conda command:

```bash
conda install pytorch torchvision -c pytorch
```

### pip
{: #mac-anaconda}

To install PyTorch via pip, use one of the following two commands, depending on your Python version:

```bash
# Python 3.x
pip3 install torch torchvision
```

## Verification
{: #mac-verification}

To ensure that PyTorch was installed correctly, we can verify the installation by running sample PyTorch code. Here we will construct a randomly initialized tensor.

```python
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

## Building from source
{: #mac-from-source}

For the majority of PyTorch users, installing from a pre-built binary via a package manager will provide the best experience. However, there are times when you may want to install the bleeding edge PyTorch code, whether for testing or actual development on the PyTorch core. To install the latest PyTorch code, you will need to [build PyTorch from source](https://github.com/pytorch/pytorch#from-source).

> You will also need to build from source if you want CUDA support.

### Prerequisites
{: #mac-prerequisites-2}

1. Install [Anaconda](#anaconda)
2. Install [CUDA](https://developer.nvidia.com/cuda-downloads), if your machine has a [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus).
3. Follow the steps described here: [https://github.com/pytorch/pytorch#from-source](https://github.com/pytorch/pytorch#from-source)

You can verify the installation as described [above](#mac-verification).
