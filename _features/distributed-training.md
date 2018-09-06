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
---

Take advantage of native support for asynchronous execution of collective operations and peer-to-peer communication that is accessible from both Python and C++.
