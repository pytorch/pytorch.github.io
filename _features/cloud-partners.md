---
title: Cloud Partners
order: 7
snippet: >
  ```sh
    export IMAGE_FAMILY="pytorch-latest-cpu"
    export ZONE="us-west1-b"
    export INSTANCE_NAME="my-instance"
    
    gcloud compute instances create $INSTANCE_NAME \
      --zone=$ZONE \
      --image-family=$IMAGE_FAMILY \
      --image-project=deeplearning-platform-release
  ```

summary-home: PyTorch is well supported on major cloud platforms, providing frictionless development and easy scaling.

---

PyTorch is well supported on major cloud platforms, providing frictionless development and easy scaling through prebuilt images, large scale training on GPUs, ability to run models in a production scale environment, and more.
