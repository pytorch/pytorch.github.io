---
title: TorchServe
order: 2
snippet: >
  ```python
    ## Convert the model from PyTorch to TorchServe format
    torch-model-archiver --model-name densenet161 \
    --version 1.0 --model-file serve/examples/image_classifier/densenet_161/model.py \
    --serialized-file densenet161-8d451a50.pth \
    --extra-files serve/examples/image_classifier/index_to_name.json \
    --handler image_classifier

    ## Host your PyTorch model

   torchserve --start --model-store model_store --models densenet161=densenet161.mar
  ```

summary-home: TorchServe is an easy to use tool for deploying PyTorch models at scale. It is cloud and environment agnostic and supports features such as multi-model serving, logging, metrics and the creation of RESTful endpoints for application integration.
featured-home: false

---

TorchServe is an easy to use tool for deploying PyTorch models at scale. It is cloud and environment agnostic and supports features such as multi-model serving, logging, metrics and the creation of RESTful endpoints for application integration.
