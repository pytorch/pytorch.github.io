---
title: Hybrid Front-End
order: 1
snippet: >
  ```python
    @torch.jit.script

    def fizzbuzz(niter : int

      for i in range(niter):
        if i % 3 == 0 and i % 5 == 0:
          print('fizzbuzz')
        elif i % 3 == 0:
          print('fizz')
        elif i % 5 == 0:
          print('buzz')
        else:
          print(i)
  ```

summary-home: A new hybrid front-end seamlessly transitions between eager mode and graph mode to provide both flexibility and speed.
featured-home: true

---

A new hybrid front-end provides ease-of-use and flexibility in eager mode, while seamlessly transitioning to graph mode for speed, optimization, and functionality in C++ runtime environments.
