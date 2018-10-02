---
title: Native ONNX Support
order: 5
snippet: >
  ```python
    import torch.onnx
    import torchvision

    dummy_input = torch.randn(1, 3, 224, 224)
    model = torchvision.models.alexnet(pretrained=True)
    torch.onnx.export(model, dummy_input, "alexnet.onnx")
  ```
---

Export models in the standard ONNX (Open Neural Network Exchange) format for direct access to ONNX-compatible platforms, runtimes, visualizers, and more.
