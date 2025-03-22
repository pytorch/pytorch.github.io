---
title: Production Ready
order: 1
snippet: >
  ```python
    import torch
    class MyModule(torch.nn.Module):

      def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

      def forward(self, input):
        if input.sum() > 0:
          output = self.weight.mv(input)
        else:
          output = self.weight + input
        return output

    # Compile the model code to a static representation
    my_script_module = torch.jit.script(MyModule(3, 4))

    # Save the compiled code and model data so it can be loaded elsewhere
    my_script_module.save("my_script_module.pt")
  ```

summary-home: Transition seamlessly between eager and graph modes with TorchScript, and accelerate the path to production with TorchServe.
featured-home: true

---

With TorchScript, PyTorch provides ease-of-use and flexibility in eager mode, while seamlessly transitioning to graph mode for speed, optimization, and functionality in C++ runtime environments.
