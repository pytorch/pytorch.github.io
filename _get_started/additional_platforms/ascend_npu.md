# Installing on Ascend NPU

Ascend NPU is Huawei's AI processor series. The Ascend Extension for PyTorch (`torch_npu`) enables PyTorch to run on Ascend NPUs, supporting eager-mode execution, distributed training, and mixed precision via the `PrivateUse1` backend.

## Prerequisites

### Hardware Requirements

* Huawei Ascend NPU(s), for example, 910B, 910C, or 310P processor

### Software Requirements

* Python >= 3.10
* CANN (Compute Architecture for Neural Networks) toolkit installed >= 8.0
* Ascend driver and firmware installed

Before installing PyTorch with Ascend NPU support, you must install the CANN toolkit. Download it from the [Ascend Community](https://www.hiascend.com/) and follow the [CANN Installation Guide](https://www.hiascend.com/cann/download?versionId=735&ids=d806%2Ch0501%2Ch0601%2Ch0702).

## Installation

### pip

Use the pip package manager to install PyTorch with Ascend NPU support. Please refer to the [installation page](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=167&ids=26958bcc909e4cd48fa56d4c4a43ebec%2C98%2C106%2C1%2C6%2C3%2C) and select your preferred options in the selector above to get the installation command.

## Verification

To ensure that PyTorch was installed correctly with Ascend NPU support, run the following code:

```python
import torch
import torch_npu

print(torch.__version__)
print("torch_npu version:", torch_npu.__version__)

if torch.npu.is_available():
    print("Ascend NPU is available!")
    print(f"NPU count: {torch_npu.npu.device_count()}")
    print(f"NPU name: {torch_npu.npu.get_device_name(0)}")

    # Test basic computation
    a = torch.randn(3, 4).npu()
    b = torch.randn(3, 4).npu()
    print("NPU computation result:", a + b)
else:
    print("Ascend NPU is not available.")
```

Expected output should show the PyTorch and torch_npu versions, NPU count, device name, and the result of a tensor addition without errors.

## Documentation

For more information, please visit:

* [Ascend Extension for PyTorch Official Documentation](https://www.hiascend.com/document/detail/zh/Pytorch/2600/index/index.html)
* [Ascend Community Portal](https://www.hiascend.com/)
* [Ascend/pytorch GitHub Repository](https://github.com/Ascend/pytorch)
* [PyPI: torch-npu](https://pypi.org/project/torch-npu/)
