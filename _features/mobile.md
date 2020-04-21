---
title: Mobile (Experimental)
order: 4
snippet: >
  ```python
    ## Save your model
    torch.jit.script(model).save("my_mobile_model.pt")

    ## iOS prebuilt binary
    pod ‘LibTorch’
    ## Android prebuilt binary
    implementation 'org.pytorch:pytorch_android:1.3.0'

    ## Run your model (Android example)
    Tensor input = Tensor.fromBlob(data, new long[]{1, data.length});
    IValue output = module.forward(IValue.tensor(input));
    float[] scores = output.getTensor().getDataAsFloatArray();
  ```

summary-home: PyTorch supports an end-to-end workflow from Python to deployment on iOS and Android. It extends the PyTorch API to cover common preprocessing and integration tasks needed for incorporating ML in mobile applications.
featured-home: false

---

PyTorch supports an end-to-end workflow from Python to deployment on iOS and Android. It extends the PyTorch API to cover common preprocessing and integration tasks needed for incorporating ML in mobile applications.
