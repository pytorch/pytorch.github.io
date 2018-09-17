---
title: Python-First
order: 3
snippet: >
  ```python
    import torch
    import numpy as np
    a = np.ones(5)
    b = torch.from_numpy(a)
    np.add(a, 1, out=a)
    print(a)
    print(b)
  ```

summary-home: Deep integration into Python allows popular libraries and packages to be used for easily writing neural network layers in Python.
featured-home: true

---

PyTorch is not a Python binding into a monolithic C++ framework. It's built to be deeply integrated into Python so it can be used with popular libraries and packages such as Cython and Numba.
