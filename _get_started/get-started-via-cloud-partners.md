---
layout: default
title: Start Via Cloud Partners
order: 3
published: true
---

## Start Via Cloud Partners

<div class="container-fluid quick-start-module quick-starts">
  <div class="cloud-options-col">
    {% include quick_start_cloud_options.html %}
  </div>
</div>

---

{% capture aws %}{% include_relative installation/aws.md %}{% endcapture %}

<div id="cloud">
  <div class="platform aws">{{aws | markdownify }}</div>
</div>