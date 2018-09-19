---
layout: get_started
title: Start Locally
permalink: /get-started/locally/
background-class: get-started-background
body-class: get-started
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
<nav class="inline_toc" markdown="1">
* TOC
{:toc}
{::options toc_levels="1..3" /}
</nav>
{% include_relative installation/mac.md %}
{% endcapture %}

{% capture linux %}
<nav class="inline_toc" markdown="1">
* TOC
{:toc}
{::options toc_levels="1..3" /}
</nav>
{% include_relative installation/linux.md %}
{% endcapture %}

{% capture windows %}
<nav class="inline_toc" markdown="1">
* TOC
{:toc}
{::options toc_levels="1..3" /}
</nav>
{% include_relative installation/windows.md %}
{% endcapture %}


<div id="installation">
  <div class="os macos">{{ mac | markdownify }}</div>
  <div class="os linux selected">{{ linux | markdownify }}</div>
  <div class="os windows">{{ windows | markdownify }}</div>
</div>

<script type="text/javascript">
  var pageId = "get-started-locally"; // TBD: Make this programmatic
  $(".main-content-menu .nav-item").removeClass("nav-select");
  $(".main-content-menu .nav-link[data-id='" + pageId + "']").parent(".nav-item").addClass("nav-select");
</script>
<script src="{{ site.baseurl }}/assets/quick-start-module.js"></script>
<script src="{{ site.baseurl }}/assets/show-screencast.js"></script>