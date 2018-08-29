# PyTorch Installation: Mac

PyTorch can be installed and used on MacOS. Depending on your system and compute requirements, your experience with PyTorch on a Mac may vary in terms of processing time. It is recommended, but not required, that your Mac have an NVIDIA GPU in order to harness the full power of PyTorch's CUDA[LINK] support.


> Currently, CUDA support on MacOS is only available by building PyTorch from source[LINK TO BELOW]

## Prerequisites

### Mac OS Version

PyTorch is supported on MacOS 10.12 (Sierra) or above.

### Python

By default, MacOS is installed with Python 2.7. PyTorch can be installed with Python 2.7, but it is recommended that you use Python 3.6 or greater, which can be installed either through the Anaconda package manager (see below[LINK]), [Homebrew](https://brew.sh/), or the [Python website](https://www.python.org/downloads/mac-osx/).

### Package Manager

To install the PyTorch binaries, you will need to use one of two supported package managers: [Anaconda](https://www.anaconda.com/download/#macos) or [pip](https://pypi.org/project/pip/). Anaconda is the recommended package manager as it will provide you all of the PyTorch dependencies in one, sandboxed install, including Python.

_Anaconda_

To install Anaconda, you can [download graphical installer](https://www.anaconda.com/download/#macos) or use the command-line installer. If you use the command-line installer, you can right-click on the installer link, select `Copy Link Address`, and then use the following commands:

```
`# The version of Anaconda may be different depending on when you are installing`
`curl -O https://repo.anaconda.com/archive/Anaconda3-5.2.0-MacOSX-x86_64.sh`
`sh Anaconda3-5.2.0-MacOSX-x86_64.sh`
`# and follow the prompts. The defaults are generally good.`
```


https://asciinema.org/a/FGxea7yIcAhTZ6rSYJ5L4Zhyd

_pip_

*Python 3*

If you installed Python via Homebrew or the Python website, `pip` was installed with it. If you installed Python 3.x, then you will be using the command `pip3`.


> Tip: If you want to use just the command  `pip`, instead of `pip3`, you can symlink `pip` to the `pip3` binary.


*Python 2*

If you are using the default installed Python 2.7, you will need to install `pip` via `easy_install`

```
`sudo easy_install pip`
```



## Installation

### Anaconda

To install PyTorch via Anaconda, use the following conda command:

```
`conda install pytorch torchvision -c pytorch` 
```


https://asciinema.org/a/knNwUMskEyLxOUV4bO9NMG3w9

### pip

To install PyTorch via pip, use one of the following two commands, depending on your Python version:

```
`# Python 3.x`
`pip3 install torch torchvision` 
```

```
`# Python 2.x`
`pip install torch torchvision `
```

## Verification

To ensure that PyTorch was installed correctly, we can verify the installation by running sample PyTorch code. Here we will construct a randomly initialized tensor.


```
**from** **__future__** **import** print_function
**import** **torch
x = torch.rand(5, 3)
print(x)**
```

The output should be something similar to:


```
tensor([[0.3380, 0.3845, 0.3217],
        [0.8337, 0.9050, 0.2650],
        [0.2979, 0.7141, 0.9069],
        [0.1449, 0.1132, 0.1375],
        [0.4675, 0.3947, 0.1426]])
```

https://asciinema.org/a/GmbKvZK3zEErwzO9sYFkwzSTJ

## Building from source

For the majority of PyTorch users, installing from a pre-built binary via a package manager will be provide the best experience. However, there are times when you may want to install the bleeding edge PyTorch code, whether for testing or actual development on the PyTorch core. To install the latest PyTorch code, you will need to [build PyTorch from source](https://github.com/pytorch/pytorch#from-source).


> You will also need to build from source if you want CUDA support.

### Prerequisites

1. Install Anaconda[LINK ABOVE]
2. Install [CUDA](https://developer.nvidia.com/cuda-downloads), if your machine has a [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus).
3. Install optional dependencies:

```
export CMAKE_PREFIX_PATH=[anaconda root directory]
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
```

### Build

```
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install
```

You can verify the installation as described above[LINK]