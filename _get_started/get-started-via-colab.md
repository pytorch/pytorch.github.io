---
layout: get_started
title: Try Now via CoLab
permalink: /get-started/colab/
background-class: get-started-background
body-class: get-started
order: 3
---

## Try Now via CoLab

Lorem ipsum dolor sit amet, ex mei graeco alienum imperdiet. Recusabo consequuntur mei ei, habeo iriure virtute eam cu, in erat placerat vis. Eu mea nostrum inimicus, cum id aeque utamur erroribus.

Lorem ipsum dolor sit amet, ex mei graeco alienum imperdiet. Recusabo consequuntur mei ei, habeo iriure virtute eam cu, in erat placerat vis. Eu mea nostrum inimicus, cum id aeque utamur erroribus.

{% highlight python %}
#!/usr/bin/python3

# Print the contents of the files listed on the command line.

import sys

for fn in sys.argv[1:]:
    try:
        fin = open(fn, 'r')
    except:
        (type, detail) = sys.exc_info()[:2]
        print("\n*** %s: %s: %s ***" % (fn, type, detail))
        continue
    print("\n*** Contents of", fn, "***")

    # Print the file, with line numbers.
    lno = 1
    while 1:
        line = fin.readline()
        if not line: break;
        print('%3d: %-s' % (lno, line[:-1]))
        lno = lno + 1
    fin.close()
print()
{% endhighlight %}

Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.


<script type="text/javascript">
  var pageId = "get-started-via-colab"; // TBD: Make this programmatic
  $(".main-content-menu .nav-item").removeClass("nav-select");
  $(".main-content-menu .nav-link[data-id='" + pageId + "']").parent(".nav-item").addClass("nav-select");
</script>
<script src="{{ site.baseurl }}/assets/quick-start-module.js"></script>
<script src="{{ site.baseurl }}/assets/show-screencast.js"></script>