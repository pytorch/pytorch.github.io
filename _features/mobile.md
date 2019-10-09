---
title: Mobile (Experimental)
order: 3
snippet: >
  ```python
    ## Author your model in PyTorch

    ## Optional Optimization
    qmodel = quantization.convert(my_mobile_model)

    ## Serialize and save your model
    torch.jit.script(qmodel).save("my_mobile_model.pt")

    ## iOS prebuilt binary
    pod ‘LibTorch’
    ## Android prebuilt binary
    implementation 'org.pytorch:pytorch_android:1.3.0'

  ```

summary-home: PyTorch supports an end-to-end workflow from Python to deployment on iOS and Android. It extends the PyTorch API to cover common preprocessing and integration tasks needed for incorporating ML in mobile applications.
featured-home: false

---

PyTorch supports an end-to-end workflow from Python to deployment on iOS and Android. It extends the PyTorch API to cover common preprocessing and integration tasks needed for incorporating ML in mobile applications.
