# Installing on Ascend NPU

**Ascend for PyTorch (TorchNPU)** is a deep learning adaptation framework built on Ascend. It enables Ascend NPUs to support the PyTorch framework, delivering the powerful computing capabilities of Ascend AI processors to PyTorch developers and users.

## Prerequisites

### Hardware Requirements

* Ascend for PyTorch supports most Ascend platforms. Please check the hardware compatibility before installing.

### Software Requirements

* Python >= 3.10
* CANN (Compute Architecture for Neural Networks) toolkit installed >= 8.0
* Ascend driver and firmware installed

Before installing PyTorch with Ascend NPU support, you must install the CANN toolkit. Download it from the [Ascend Community](https://www.hiascend.com/en) and follow the [CANN Installation Guide](https://www.hiascend.com/en/cann/download).

## Installation

### pip

```bash
pip3 install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu && pip3 install torch-npu==2.10.0
```

Use the pip package manager to install PyTorch with Ascend NPU support. Please refer to the [installation page](https://www.hiascend.com/en/developer/software/ai-frameworks/pytorch/download) and select your preferred options in the selector above to get the installation command.

## Verification

To ensure that PyTorch was installed correctly with Ascend NPU support, run the following code:

```python
import torch
import torch_npu

x = torch.randn(2, 2, device="npu")
y = torch.randn(2, 2, device="npu")
z = x.mm(y)

print(z)
```

The following, or a similar output, indicates successful installation:

```bash
tensor([[-0.0515,  0.3664],
        [-0.1258, -0.5425]],  device='npu:0')
```

## Documentation

For more information, please visit:

* Ascend for PyTorch Official Documentation ([English](https://www.hiascend.com/document/detail/en/Pytorch/latest/index/index.html), [Chinese](https://www.hiascend.com/document/detail/zh/Pytorch/2600/index/index.html))
* Ascend Community Portal ([English](https://www.hiascend.com/en), [Chinese](https://www.hiascend.com/))
* CANN Main Page ([English](https://www.hiascend.com/eng/cann), [Chinese](https://www.hiascend.com/cann))
* PyTorch Adaptation for Ascend Portal ([English](https://www.hiascend.com/en/developer/software/ai-frameworks/pytorch), [Chinese](https://www.hiascend.com/cn/developer/software/ai-frameworks/pytorch))
* Ascend Hardware Compatibility (English *coming soon*, [Chinese](https://www.hiascend.com/hardware/compatibility))
* [GitHub Repository](https://github.com/Ascend/pytorch)
* [PyPI](https://pypi.org/project/torch-npu/)
