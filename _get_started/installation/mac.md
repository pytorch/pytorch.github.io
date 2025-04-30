# Installing on macOS
{:.no_toc}

PyTorch can be installed and used on macOS. Depending on your system and GPU capabilities, your experience with PyTorch on a Mac may vary in terms of processing time.

## Prerequisites
{: #mac-prerequisites}

### macOS Version

PyTorch is supported on macOS 10.15 (Catalina) or above.

### Python
{: #mac-python}

It is recommended that you use Python 3.9 - 3.12.
You can install Python either through [Homebrew](https://brew.sh/) or
the [Python website](https://www.python.org/downloads/mac-osx/).

### Package Manager
{: #mac-package-manager}

To install the PyTorch binaries, you will need to use the supported package manager: [pip](https://pypi.org/project/pip/).
#### pip

*Python 3*

If you installed Python via Homebrew or the Python website, `pip` was installed with it. If you installed Python 3.x, then you will be using the command `pip3`.

> Tip: If you want to use just the command  `pip`, instead of `pip3`, you can symlink `pip` to the `pip3` binary.

## Installation
{: #mac-installation}

### pip
{: #mac-pip}

To install PyTorch via pip, use the following the command, depending on your Python version:

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

### Prerequisites
{: #mac-prerequisites-2}

1. [Optional] Install [pip](https://pypi.org/project/pip/)
2. Follow the steps described here: [https://github.com/pytorch/pytorch#from-source](https://github.com/pytorch/pytorch#from-source)

You can verify the installation as described [above](#mac-verification).
