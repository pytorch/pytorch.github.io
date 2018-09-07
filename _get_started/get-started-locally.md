---
layout: default
title: Start Locally
order: 1
published: true
---

## Start Locally

<div class="container-fluid quick-start-module quick-starts">
  <div class="row">
    <div class="col-md-12">
      {% include quick_start_local.html %}
    </div>
  </div>
</div>

---

{% capture mac %}
<div class="inline_toc" markdown="1">
* TOC
{:toc}
</div>
{% include_relative installation/mac.md %}
{% endcapture %}

{% capture linux %}
<div class="inline_toc" markdown="1">
* TOC
{:toc}
</div>
{% include_relative installation/linux.md %}
{% endcapture %}

{% capture windows %}
<div class="inline_toc" markdown="1">
* TOC
{:toc}
</div>
{% include_relative installation/windows.md %}
{% endcapture %}


<div id="installation">
  <div class="os macos">{{ mac | markdownify }}</div>
  <div class="os linux selected">{{ linux | markdownify }}</div>
  <div class="os windows">{{ windows | markdownify }}</div>
</div>
