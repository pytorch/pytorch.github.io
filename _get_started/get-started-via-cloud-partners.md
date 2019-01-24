---
layout: get_started
title: Start via Cloud Partners
permalink: /get-started/cloud-partners/
background-class: get-started-background
body-class: get-started
order: 2
published: true
---

## Start via Cloud Partners

<div class="container-fluid quick-start-module quick-starts">
  <div class="cloud-options-col">
    <p>Cloud platforms provide powerful hardware and infrastructure for training and deploying deep learning models. Select a cloud platform below to get started with PyTorch.</p>
    {% include quick_start_cloud_options.html %}
  </div>
</div>

---

{% capture aws %}
<nav class="inline_toc" markdown="1">
* TOC
{:toc}
</nav>
{% include_relative installation/aws.md %}
{% endcapture %}

{% capture azure %}
<nav class="inline_toc" markdown="1">
* TOC
{:toc}
</nav>
{% include_relative installation/azure.md %}
{% endcapture %}

{% capture google-cloud %}
<nav class="inline_toc" markdown="1">
* TOC
{:toc}
</nav>
{% include_relative installation/google-cloud.md %}
{% endcapture %}

{% capture floydhub %}
<nav class="inline_toc" markdown="1">
* TOC
{:toc}
</nav>
{% include_relative installation/floydhub.md %}
{% endcapture %}


<div id="cloud">
  <div class="platform aws">{{aws | markdownify }}</div>
  <div class="platform google-cloud">{{google-cloud | markdownify }}</div>
  <div class="platform microsoft-azure">{{azure | markdownify }}</div>
  <div class="platform floydhub">{{floydhub | markdownify }}</div>
</div>

<script type="text/javascript">
  var pageId = "get-started-via-cloud-partners"; // TBD: Make this programmatic
  $(".main-content-menu .nav-item").removeClass("nav-select");
  $(".main-content-menu .nav-link[data-id='" + pageId + "']").parent(".nav-item").addClass("nav-select");
</script>
<script src="{{ site.baseurl }}/assets/quick-start-module.js"></script>
<script src="{{ site.baseurl }}/assets/show-screencast.js"></script>