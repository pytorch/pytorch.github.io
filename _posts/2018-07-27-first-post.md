---
layout: default
title: First Post
category: [blog]
---

# First Post

This is a demo of all styled elements in Jekyll.

This is a paragraph, it's surrounded by whitespace. Next up are some headers, they're heavily influenced by GitHub's markdown style.

## Header 2 (H1 is reserved for post titles)##

### Header 3

#### Header 4

A link to [Pytorch](https://pytorch.org). A literal link <https://pytorch.org>

An image, located within /assets/images

![an image alt text]({{ site.baseurl }}/assets/images/sample.jpg "an image title")

* A bulletted list
- alternative syntax 1
+ alternative syntax 2
  - an indented list item

1. An
2. ordered
3. list

Inline markup styles:

- _italics_
- **bold**
- `code()`

> Blockquote
>> Nested Blockquote

Syntax highlighting can be used by wrapping your code in a liquid tag like so:

{{ "{% highlight javascript " }}%}
/* Some pointless Javascript */
var test = ["t", "e", "s", "t"];
{{ "{% endhighlight " }}%}

creates...

{% highlight javascript %}
/* Some pointless Javascript */
var test = ["t", "e", "s", "t"];
{% endhighlight %}

Use two trailing spaces
on the right
to create linebreak tags

Finally, horizontal lines

----
****
