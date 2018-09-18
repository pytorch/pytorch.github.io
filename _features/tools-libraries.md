---
title: Tools & Libraries
order: 4
snippet: >
  ```python
    from torchvision import transforms, utils
    
    scale = Rescale(256)
    crop = RandomCrop(128)
    composed = transforms.Compose([Rescale(256),
    RandomCrop(224)])

    fig = plt.figure()
    sample = face_dataset[65]
    for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

    plt.show()
  ```

summary-home: A rich ecosystem of tools and libraries extends PyTorch and supports development in computer vision, NLP and more.
featured-home: true

---

An active community of researchers and developers have built a rich ecosystem of tools and libraries for extending PyTorch and supporting development in areas from computer vision to reinforcement learning.
