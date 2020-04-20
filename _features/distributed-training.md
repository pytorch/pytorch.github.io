---
title: Distributed Training
order: 3
snippet: >
  ```python
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    
    dist.init_process_group(backend='gloo')
    model = DistributedDataParallel(model)
  ```

summary-home: Scalable distributed training and performance optimization in research and production is enabled by the torch.distributed backend.
featured-home: true

---

Optimize performance in both research and production by taking advantage of native support for asynchronous execution of collective operations and peer-to-peer communication that is accessible from Python and C++.
