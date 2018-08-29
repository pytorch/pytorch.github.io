---
layout: default
title: Start Locally
slug: start-locally
category: [get-started]
date: 2018-07-31 00:00:01
---

{% include quick_start_locally_module.html %}

---

{% capture mac %}{% include_relative installation/mac.md %}{% endcapture %}
{% capture linux %}{% include_relative installation/linux.md %}{% endcapture %}
{% capture windows %}{% include_relative installation/windows.md %}{% endcapture %}

<div id="installation">
  <div class="os macos">{{mac | markdownify }}</div>
  <div class="os linux selected">{{linux | markdownify }}</div>
  <div class="os windows">{{windows | markdownify }}</div>
</div>
