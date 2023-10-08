---
layout: executorch
title: Documentation
permalink: /executorch/documentation/
background-class: mobile-background
body-class: mobile
order: 2
published: true
---

# Welcome to ExecuTorch Documentation

**IMPORTANT NOTE: This is a preview version of Executorch and should be used for testing and evaluation purposes only. It is not recommended for use in production settings. We welcome any feedback, suggestions, and bug reports from the community to help us improve the technology.**

[ExecuTorch](/executorch/home) is a PyTorch platform that provides infrastructure to run PyTorch programs everywhere from AR/VR wearables to standard on-device iOS and Android mobile deployments. One of the main goals for ExecuTorch is to extend PyTorch’s customization and deployment capabilities to edge devices, including embedded.

ExecuTorch heavily relies on such PyTorch technologies as TorchDynamo and torch.export. If you are not familiar with these APIs, you might want to read about them in the PyTorch documentation before diving into the ExecuTorch documentation. You may also check out the [ExecuTorch Concepts](https://docs.google.com/document/d/1eKsC-YZrvIsJwRmY6-mI5zVCsGKxm1r3_kYW4PnUtv4/edit) page to learn more about relevant background and key concepts related to ExecuTorch. 

ExecuTorch is still in preview mode and is being actively developed. We are releasing it early to the community as we love to improve and expand its capabilities with you. This means you’re welcome to download and try ExecuTorch and provide us with your valuable feedback while being aware that the APIs might change. Please use the [PyTorch Forums](https://discuss.pytorch.org/) for discussion and feedback about ExecuTorch and our GitHub page for bug reporting using the tag #executorch.

Features described in this documentation are classified by release status:

- _Stable:_ These features will be maintained long-term and there should generally be no major performance limitations or gaps in documentation. We also expect to maintain backwards compatibility (although breaking changes can happen and notice will be given one release ahead of time).
- _Beta:_ These features are tagged as Beta because the API may change based on user feedback, because the performance needs to improve, or because coverage across operators is not yet complete. For Beta features, we are committing to seeing the feature through to the Stable classification. We are not, however, committing to backwards compatibility.
-  _Prototype:_ These features are typically not available as part of binary distributions like PyPI or Conda, except sometimes behind run-time flags, and are at an early stage for feedback and testing.


<!-- Do not remove the below script -->

<script page-id="documentation" src="{{ site.baseurl }}/assets/menu-tab-selection.js"></script>
