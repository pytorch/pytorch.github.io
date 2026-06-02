# Installing on Intel GPU (XPU) Platform

XPU is a PyTorch device backend designed to support hardware acceleration on Intel GPUs. Key technical features:

* Native support for FP32, BF16, FP16, and Automatic Mixed Precision (AMP)
* Extensions of operator set through custom SYCL kernels
* Graph compilation
* Distributed training (through `XCCL`)

## Prerequisites

### Hardware Requirements

* Intel Client GPU:

  * Intel® Arc A-Series Graphics (CodeName: Alchemist)
  * Intel® Arc B-Series Graphics (CodeName: Battlemage)
  * Intel® Core™ Ultra Processors with Intel® Arc™ Graphics (CodeName: Meteor Lake-H)
  * Intel® Core™ Ultra Processors (Series 2) with Intel® Arc™ Graphics (CodeName: Arrow Lake-H)
  * Intel® Core™ Ultra Mobile Processors (Series 2) with Intel® Arc™ Graphics (CodeName: Lunar Lake)
  * Intel® Core™ Ultra Mobile Processors (Series 3) with Intel® Arc™ Graphics (CodeName: Panther Lake)

* Intel Data Center GPU:

  * Intel® Data Center GPU Max Series (CodeName: Ponte Vecchio)

### Software Requirements

* [Intel GPU Driver](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu.html)
* Python 3.10 or later

## Installation

### pip

Use the pip package manager to install PyTorch with XPU support. Select your preferred options in the selector above to get the installation command.

## Verification

To ensure that PyTorch was installed correctly with XPU support, run the following code:

```python
import torch
print(torch.__version__)

# Check XPU availability
if torch.xpu.is_available():
    print("XPU is available!")
    print(f"XPU devices: {torch.xpu.device_count()}")
else:
    print("XPU is not available.")
```

## Documentation

For more information, please visit the [Getting Started on Intel GPU](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html).
