---
title: Distributed Training
order: 2
snippet: >
  ```python
    #!/usr/bin/python3

    # Simple while loop
    a = 0
    while a < 15:
        print(a, end=' ')
        if a == 10:
            print("made it to ten!!")
        a = a + 1
    print()
  ```

summary-home: Optimize performance with scalable distributed training in both research and production through the torch.distributed backend.
featured-home: true

---

Optimize performance in both research and production by taking advantage of native support for asynchronous execution of collective operations and peer-to-peer communication that is accessible from Python and C++.
