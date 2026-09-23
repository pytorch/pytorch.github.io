# Installing on Google Cloud TPU (TorchTPU)

**TorchTPU (`torch_tpu`)** is a native PyTorch backend built for Google Cloud Tensor Processing Units (TPUs). It enables Google Cloud TPUs to run PyTorch workloads natively (`PrivateUse1` dispatch key `"tpu"`), supporting eager execution `torch.compile()`, distributed training (`torch.distributed`, `DTensor`, `FSDP2`), and custom kernels (`Pallas`, `Helion`).

## Prerequisites

### Hardware Requirements

* A provisioned Google Cloud TPU VM (TPU v5e, v5p, v6e, or v7x) on Google Compute Engine (GCE) or Google Kubernetes Engine (GKE), or a TPU runtime in Google Colab.

### Software Requirements

* Linux (Ubuntu 22.04+ recommended)
* Python >= 3.10
* `libtpu` runtime library (automatically installed with `torch-tpu`)
* **Recommended Memory Allocator**: For optimal runtime performance, configure `TCMalloc` (`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4`) prior to launching Python workloads, following the [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#switch-memory-allocator).

## Installation

### pip

```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cpu && pip3 install torch-tpu
```

Use the `pip` package manager to install the CPU build of PyTorch alongside `torch-tpu`. Select your preferred options in the selector above to get the installation command.

## Verification

To ensure that PyTorch was installed correctly with TorchTPU support, run the following code:

```python
import torch
device =torch.device("tpu")

x = torch.randn(2, 2, device="tpu")
y = torch.randn(2, 2, device="tpu")
z = x.mm(y)

print(z)
```

The following, or a similar output, indicates successful installation:

```bash
tensor([[-0.4218,  0.8912],
        [ 0.1534, -1.1045]], device='tpu:0')
```

## Documentation

For more information, please visit:

* [TorchTPU Official Documentation & User Guide](https://google-pytorch.github.io/torch_tpu/)
* [Supported vs. Unsupported Feature Matrix](https://github.com/google-pytorch/torch_tpu/blob/main/docs/features_matrix.md)
* [Google Cloud TPU Documentation](https://cloud.google.com/tpu/docs)
* [GitHub Repository](https://github.com/google-pytorch/torch_tpu)
* [PyPI](https://pypi.org/project/torch-tpu/)
