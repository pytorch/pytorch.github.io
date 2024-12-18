---
layout: blog_detail
title: "docTR joins PyTorch Ecosystem: From Pixels to Data, Building a Recognition Pipeline with PyTorch and docTR"
author: Olivier Dulcy & Sebastian Olivera, Mindee
hidden: true
---

![docTR logo](/assets/images/doctr-joins-pytorch-ecosystem/fg1.png){:style="width:100%;display: block;max-width:400px; margin-left:auto; margin-right:auto;"}

We’re thrilled to announce that the docTR project has been integrated into the PyTorch ecosystem! This integration ensures that docTR aligns with PyTorch’s standards and practices, giving developers a reliable, community-backed solution for powerful OCR workflows.

**For more information on what it means to be a PyTorch ecosystem project, see the [PyTorch Ecosystem Tools page](https://pytorch.org/ecosystem/).**


## About docTR

docTR is an Apache 2.0 project developed and distributed by [Mindee](https://www.mindee.com/) to help developers integrate OCR capabilities into applications with no prior knowledge required.

To quickly and efficiently extract text information, docTR uses a two-stage approach:



* First, it performs text **detection** to localize words.
* Then, it conducts text **recognition** to identify all characters in a word.

**Detection** and **recognition** are performed by state-of-the-art models written in PyTorch. To learn more about this approach, you can refer [to the docTR documentation](https://mindee.github.io/doctr/using_doctr/using_models.html).

docTR enhances the user experience in PyTorch projects by providing high-performance OCR capabilities right out of the box. Its specially designed models require minimal to no fine-tuning for common use cases, allowing developers to quickly integrate advanced document analysis features.


## Local installation

docTR requires Python >= 3.10 and supports Windows, Mac and Linux. Please refer to our [README](https://github.com/mindee/doctr?tab=readme-ov-file#installation) for necessary dependencies for MacBook with the M1 chip.

```
pip3 install -U pip
pip3 install "python-doctr[torch,viz]"
```

This will install docTR along with the latest version of PyTorch.


```
Note: docTR also provides docker images for an easy deployment, such as a part of Kubernetes cluster.
```



## Text recognition

Now, let’s try docTR’s OCR recognition on this sample:


![OCR sample](/assets/images/doctr-joins-pytorch-ecosystem/fg2.jpg){:style="width:100%;display: block;max-width:300px; margin-left:auto; margin-right:auto;"}


The OCR recognition model expects an image with only one word on it and will output the predicted word with a confidence score. You can use the following snippet to test OCR capabilities from docTR:

```
python
from doctr.io import DocumentFile
from doctr.models import recognition_predictor

doc = DocumentFile.from_images("/path/to/image")

# Load the OCR model
# This will download pre-trained models hosted by Mindee
model = recognition_predictor(pretrained=True)

result = model(doc)
print(result)
```

Here, the most important line of code is `model = recognition_predictor(pretrained=True)`. This will load a default text recognition model, `crnn_vgg16_bn`, but you can select other models through the `arch` parameter. You can check out the [available architectures](https://mindee.github.io/doctr/using_doctr/using_models.html).

When run on the sample, the recognition predictor retrieves the following data: `[('MAGAZINE', 0.9872216582298279)]`


```
Note: using the DocumentFile object docTR provides an easy way to manipulate PDF or Images.
```



## Text detection

The last example was a crop on a single word. Now, what about an image with several words on it, like this one?


![photo of magazines](/assets/images/doctr-joins-pytorch-ecosystem/fg3.jpg){:style="width:100%;display: block;max-width:300px; margin-left:auto; margin-right:auto;"}


A text detection model is used before the text recognition to output a segmentation map representing the location of the text. Following that, the text recognition is applied on every detected patch.

Below is a snippet to run only the detection part:

```
from doctr.io import DocumentFile
from doctr.models import detection_predictor
from matplotlib import pyplot as plt
from doctr.utils.geometry import detach_scores
from doctr.utils.visualization import draw_boxes

doc = DocumentFile.from_images("path/to/my/file")
model = detection_predictor(pretrained=True)

result = model(doc)

draw_boxes(detach_scores([result[0]["words"]])[0][0], doc[0])
plt.axis('off')
plt.show()
```

Running it on the full sample yields the following:


![photo of magazines](/assets/images/doctr-joins-pytorch-ecosystem/fg4.png){:style="width:100%;display: block;max-width:300px; margin-left:auto; margin-right:auto;"}


Similarly to the text recognition, `detection_predictor` will load a default model (`fast_base` here). You can also load another one by providing it through the `arch` parameter.


## The full implementation

Now, let’s plug both components into the same pipeline. 

Conveniently, docTR provides a wrapper that does exactly that for us:

```
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

doc = DocumentFile.from_images("/path/to/image")

model = ocr_predictor(pretrained=True, assume_straight_pages=False)

result = model(doc)
result.show()
```

![photo of magazines](/assets/images/doctr-joins-pytorch-ecosystem/fg5.png){:style="width:100%;display: block;max-width:300px; margin-left:auto; margin-right:auto;"}

The last line should display a matplotlib window which shows the detected patches. Hovering the mouse over them will display their contents.

You can also do more with this output, such as reconstituting a synthetic document like so:

```
import matplotlib.pyplot as plt

synthetic_pages = result.synthesize()
plt.imshow(synthetic_pages[0])
plt.axis('off')
plt.show()
```

![black text on white](/assets/images/doctr-joins-pytorch-ecosystem/fg6.png){:style="width:100%;display: block;max-width:300px; margin-left:auto; margin-right:auto;"}


The pipeline is highly customizable, where you can modify the detection or recognition model behaviors by passing arguments to the `ocr_predictor`. Please refer to the [documentation](https://mindee.github.io/doctr/using_doctr/using_models.html) to learn more about it. 


## Conclusion

We’re excited to welcome docTR into the PyTorch Ecosystem, where it seamlessly integrates with PyTorch pipelines to deliver state-of-the-art OCR capabilities right out of the box. 

By empowering developers to quickly extract text from images or PDFs using familiar tooling, docTR simplifies complex document analysis tasks and enhances the overall PyTorch experience.

We invite you to explore the [docTR GitHub repository](https://github.com/mindee/doctr), join the [docTR community on Slack](https://slack.mindee.com/), and reach out at contact@mindee.com for inquiries or collaboration opportunities. 

Together, we can continue to push the boundaries of document understanding and develop even more powerful, accessible tools for everyone in the PyTorch community.