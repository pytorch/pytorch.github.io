# Installing on CUDA - Windows on Arm

**CUDA - Windows on Arm** provides native Windows ARM64 PyTorch, TorchVision and TorchAudio wheels with NVIDIA CUDA support. It uses PyTorch's official in-tree CUDA backend and existing CUDA dispatch keys (it does not introduce a new `PrivateUse1` backend), so existing CUDA code runs unchanged on Windows on Arm systems with NVIDIA GPUs. NVIDIA builds pinned upstream PyTorch commits natively for ARM64, publishes stable and nightly packages, and runs the upstream CUDA test suite on self-hosted Windows ARM64 systems. Results are reported to the [PyTorch HUD](https://hud.pytorch.org/crcr/NVIDIA/pytorch-windows-ci?event=nightly).

## Prerequisites

### Hardware Requirements

* A Windows on Arm system (native ARM64) with a supported NVIDIA GPU.
* Supported GPU architectures: SM 8.9, SM 10.3 (+PTX), SM 12.0, SM 12.1 (+PTX).

### Software Requirements

* Windows 11 on Arm, 24H2 or later
* NVIDIA Windows driver R615 or later
* Python 3.13 (native ARM64 build; the current CI-validated configuration)
* CUDA 13.4 and cuDNN 9.26 runtime libraries (the CUDA 13.4 wheels are built against these versions)

## Installation

### pip

```bash
python -m pip install "torch==2.14.0+cu134" "torchvision==0.29.0+cu134" "torchaudio==2.11.0+cu134" --extra-index-url https://pypi.nvidia.com/nvtorch_oot/
```

Use the `pip` package manager to install PyTorch, TorchVision and TorchAudio with CUDA support from the NVIDIA package index. Select your preferred options in the selector above to get the installation command. Nightly builds are available from `https://pypi.nvidia.com/nvtorch_oot_nightly/`.

## Verification

To ensure that PyTorch was installed correctly with CUDA support, run the following code:

```python
import torch
print(torch.__version__)

# Check CUDA availability
if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

    x = torch.randn(2, 2, device="cuda")
    y = torch.randn(2, 2, device="cuda")
    print(x.mm(y))
else:
    print("CUDA is not available.")
```

The following, or a similar output, indicates successful installation:

```bash
2.14.0+cu134
CUDA is available!
CUDA devices: 1
Device name: NVIDIA ...
tensor([[-0.4218,  0.8912],
        [ 0.1534, -1.1045]], device='cuda:0')
```

## Support

* PyTorch issues: please file them at [pytorch/pytorch](https://github.com/pytorch/pytorch/issues) with the [`module: cuda-windows-arm64`](https://github.com/pytorch/pytorch/issues?q=label%3A%22module%3A+cuda-windows-arm64%22) label.
* Packaging, build and CI issues: [NVIDIA/pytorch-windows-ci](https://github.com/NVIDIA/pytorch-windows-ci/issues).

## Documentation

For more information, please visit:

* [Install built wheels](https://github.com/NVIDIA/pytorch-windows-ci#install-built-wheels)
* [Build PyTorch on Windows ARM64](https://github.com/NVIDIA/pytorch-windows-ci/blob/main/docs/build_pytorch_windows_arm64.md)
* [Windows on Arm CI coverage](https://github.com/NVIDIA/pytorch-windows-ci/blob/main/docs/woa-ci.md)
* [CI results on the PyTorch HUD](https://hud.pytorch.org/crcr/NVIDIA/pytorch-windows-ci?event=nightly)
* [GitHub Repository](https://github.com/NVIDIA/pytorch-windows-ci)
