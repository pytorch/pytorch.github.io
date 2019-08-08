---
title: TorchScript
order: 1
snippet: >
  ```python
    @torch.jit.script
    def RNN(h, x, W_h, U_h, W_y, b_h, b_y):
      y = []
      for t in range(x.size(0)):
        h = torch.tanh(x[t] @ W_h + h @ U_h + b_h)
        y += [torch.tanh(h @ W_y + b_y)]
        if t % 10 == 0:
          print("stats: ", h.mean(), h.var())
      return torch.stack(y), h
  ```

summary-home: TorchScript provides a seamless transition between eager mode and graph mode to accelerate the path to production.
featured-home: true

---

With TorchScript, PyTorch provides ease-of-use and flexibility in eager mode, while seamlessly transitioning to graph mode for speed, optimization, and functionality in C++ runtime environments.
