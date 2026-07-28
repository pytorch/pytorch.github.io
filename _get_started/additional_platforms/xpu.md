# Installing on Intel GPU (XPU) Platform

XPU device backend brings native Intel GPU support to PyTorch, enabling performant training and inference on both Linux and Windows.

## Prerequisites

The system with configured Intel GPU card is required. For detailed list of supported devices and driver install instructions refer to [Getting Started on Intel GPU](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html).

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

The following, or a similar output, indicates successful installation:

```bash
2.13.0+xpu
XPU is available!
XPU devices: 4
```

## Documentation

For more information, please visit the [torch.xpu](https://docs.pytorch.org/docs/stable/xpu.html).
