---
layout: blog_detail
title: 'Towards Reproducible Research with PyTorch Hub'
author: Team PyTorch
redirect_from: /2019/06/05/pytorch_hub.html
---

Reproducibility is an essential requirement for many fields of research including those based on machine learning techniques. However, many machine learning publications are either not reproducible or are difficult to reproduce. With the continued growth in the number of research publications, including tens of thousands of papers now hosted on arXiv and submissions to conferences at an all time high, research reproducibility is more important than ever. While many of these publications are accompanied by code as well as trained models which is helpful but still leaves a number of steps for users to figure out for themselves.

We are excited to announce the availability of PyTorch Hub, a simple API and workflow the provides the basic building blocks for improving machine learning research reproducibility. PyTorch Hub consists of a pre-trained model repository designed specifically to facilitate research reproducibility.

To illustrate the steps of how repo owners publish a model and how users can easily load and use the model, we are going to use a pre-trained BertForMaskedLM model contributed by HuggingFace.

<div class="text-center">
  <img src="{{ site.url }}/assets/images/Bert_HF.png" width="100%">
</div>

## [Owner] Publishing models

PyTorch Hub supports the publication of pre-trained models (model definitions and pre-trained weights) to a GitHub repository by adding a simple ```hubconf.py``` file. This provides an enumeration of which models are to be supported and a list of dependencies needed to run the models. An example can be found [here](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/hubconf.py).

```python
dependencies = ['torch', 'tqdm', 'boto3', 'requests', 'regex']

from hubconfs.bert_hubconf import (
    bertTokenizer,
    bertModel,
    bertForNextSentencePrediction,
    bertForPreTraining,
    bertForMaskedLM,
    bertForSequenceClassification,
    bertForMultipleChoice,
    bertForQuestionAnswering,
    bertForTokenClassification
)
from hubconfs.gpt_hubconf import (
    openAIGPTTokenizer,
    openAIGPTModel,
    openAIGPTLMHeadModel,
    openAIGPTDoubleHeadsModel
)
```

Each model then requires an entrypoint to be created. Here is a code snippet to specify an entrypoint of the ```bertForMaskedLM``` model, which returns the pre-trained model weights.

```python
def bertForMaskedLM(*args, **kwargs):
    """
    BertForMaskedLM includes the BertModel Transformer followed by the
    pre-trained masked language modeling head.
    Example:
      ...
    """
    model = BertForMaskedLM.from_pretrained(*args, **kwargs)
    return model
```

With these in place and a pull request based on the template [here](https://github.com/pytorch/hub/blob/master/docs/template.md) you are all set to go. We may work with you to refine your code snippet but this is only because it is used as part of continuous integration (CI) and will help to identify any breaking changes in a timely manner. Your model will soon appear on [Pytorch hub webpage](https://pytorch.org/hub) for all users to explore.


## [User] Workflow

As a user, PyTorch Hub allows you to follow a few simple steps and do things like: 1) explore available models; 2) load a model; and 3) understand what methods are available for any given model. Let's walk through some examples of each.

### Explore available entrypoints.

Users can list all available entrypoints in a repo using the ```torch.hub.list()``` API. In this case, we are exploring the HuggingFace Bert models and we can see that there are several models available for us to try out.

```python
>>> torch.hub.list('huggingface/pytorch-pretrained-BERT')
>>>
['bertForMaskedLM',
 'bertForNextSentencePrediction',
 ...
 'openAIGPTModel',
 'openAIGPTTokenizer']
 ```

Note that PyTorch Hub also allows auxillary entrypoints (other than pretrained models), e.g. ```bertTokenizer``` for preprocessing, to make the user workflow smoother.


### Load a model

Now that we know which models are available in the Hub, users can load a model entrypoint using the ```torch.hub.load()``` API. This only requires a single command without the need to install a wheel. In addition the ```torch.hub.help()``` API can provide useful information about how to instantiate the model.

```python
print(torch.hub.help('huggingface/pytorch-pretrained-BERT', 'bertForMaskedLM'))
model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertForMaskedLM', 'bert-base-cased')
```

It is also common that repo owners will want to continually add bug fixes or performance improvements. PyTorch Hub makes it super simple for users to get the latest update by calling:

```python
model = torch.hub.load(..., force_reload=True)
```

We believe this will help to alleviate the burden of repetitive package releases by repo owners and instead allow them to focus more on their research. It also ensures that, as a user, you are getting the freshest available models.


### Explore a loaded model

Once you have a model from PyTorch Hub loaded, you can use the following workflow to find out the available methods that are supported as well as understand better what arguments are requires to run it.


```dir(model)``` to see all available methods of the model. Let's take a look at bertForMaskedLM's available methods.

```python
>>> dir(model)
>>>
['forward'
...
'to'
'state_dict',
]
```

```help(model.forward)``` provides a view into what arguments are required to make your loaded model run

```python
>>> help(model.forward)
>>>
Help on method forward in module pytorch_pretrained_bert.modeling:
forward(input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None)
...
```

Note that the ```*args```, ```**kwargs``` passed to hub.load() are used to *instantiate* a model.


You can find a full script for predicting a masked word using the Bert model from PyTorch Hub [here](https://github.com/pytorch/hub/blob/master/huggingface_pytorch-pretrained-bert_bert.md).

## Resources to get started

* PyTorch Hub API documentation can be found [here](https://pytorch.org/docs/stable/hub.html).
* Submit a model [here](https://github.com/pytorch/hub) for publication in PyTorch Hub.
* Go to [https://pytorch.org/hub](https://pytorch.org/hub) to learn more about the available models.


A BIG thanks to the folks at HuggingFace, fast.ai and Nvidia as well as Morgane Riviere (FAIR Paris) and lots of others for the help to bootstrap this effort!!


## FAQ:

**Q: If we would like to contribute a model that is already in the Hub but perhaps mine has better accuracy, should I still contribute?**
A: Yes!! A next step for Hub is to implement an upvote/downvote system to surface the best models.

**Q: Who hosts the model weights for PyTorch Hub?**
A: You, as the contributor, are responsible to host the model weights. You can host your model in your favorite cloud storage or, if it fits within the limits, on GitHub.

**Q: What if my model is trained on private data? Should I still contribute this model?**
A: No! PyTorch Hub is centered around open research and that extends to the usage of open datasets to train these models on. If a pull request for a proprietary model is submitted, we will kindly ask that you resubmit a model trained on something open and available.

**Q: Where are my downloaded models saved?**
A: The locations are used in the order of:

* Calling ```hub.set_dir(<PATH_TO_HUB_DIR>)```
* ```$TORCH_HOME/hub```, if environment variable ```TORCH_HOME``` is set.
* ```$XDG_CACHE_HOME/torch/hub```, if environment variable ```XDG_CACHE_HOME``` is set.
* ```~/.cache/torch/hub```


Cheers!

Team PyTorch
