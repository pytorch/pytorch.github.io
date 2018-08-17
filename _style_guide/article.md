---
layout: default
title: Base Style Guide
---

## Header 2
This is body copy. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea.

### Header 3

This is body copy. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea.

#### Header 4

This is body copy. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea.

##### Header 5

This is body copy. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea.

---

This is more body copy with `code snippets`. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. [Here is an inline link](#). Ut enim ad minim veniam, quis nostrud `torch.*.FloatTensor` ullamco laboris nisi ut aliquip ex ea commodo consequat.

_This is italicized body copy. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat_

**This is bolded body copy. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.**

---

This is body copy before an unordered list. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea.

- Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
- Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
- Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

This is body copy after an unordered list. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea.

1. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
2. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
3. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

This is body copy after an ordered list. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea.

<dl>
  <dt>Definition list</dt>
  <dd>Lorem ipsum dolor sit amet, consectetur adipiscing elit</dd>

  <dt>Definition list</dt>
  <dd>Lorem ipsum dolor sit amet, consectetur adipiscing elit</dd>

  <dt>Definition list</dt>
  <dd>Lorem ipsum dolor sit amet, consectetur adipiscing elit</dd>
</dl>

---

![Here's an image](http://via.placeholder.com/1000x200/e44c2c/ffffff "Sample image")

---

> "This is a blockquote. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat"

```sh
  brew install pytorch # Here is a small code block
  brew install pytorch # Here is a small code block
```

```python
# Here is a large code block with syntax highlighting

# !/usr/bin/python3

# Dictionaries map keys to values.

fred = { 'mike': 456, 'bill': 399, 'sarah': 521 }

# Subscripts.
try:
    print(fred)
    print(fred['bill'])
    print(fred['nora'])
    print("Won't see this!")
except KeyError as rest:
    print("Lookup failed:", rest)
print()

# Entries can be added, udated, or deleted.
fred['bill'] = 'Sopwith Camel'
fred['wilma'] = 2233
del fred['mike']
print(fred)
print()

# Get all the keys.
print(fred.keys())
for k in fred.keys():
    print(k, "=>", fred[k])
print()

# Test for presence of a key.
for t in [ 'zingo', 'sarah', 'bill', 'wilma' ]:
    print(t,end=' ')
    if t in fred:
        print('=>', fred[t])
    else:
        print('is not present.')
```

Here is a table:

| Data | type torch.dtype | Tensor types |
|------|------------------|--------------|
| 32-bit floating point | `torch.float32` or `torch.float` | `torch.*.FloatTensor`
| 64-bit floating point | `torch.float64` or `torch.double` | `torch.*.DoubleTensor`
| 16-bit floating point | `torch.float16` or `torch.half` | `torch.*.HalfTensor`
| 8-bit integer (unsigned) |  `torch.uint8` | `torch.*.ByteTensor`
| 8-bit integer (signed) | `torch.int8` | `torch.*.CharTensor`
| 16-bit integer (signed) | `torch.int16` or `torch.short` | `torch.*.ShortTensor`
| 32-bit integer (signed) | `torch.int32` or `torch.int` | `torch.*.IntTensor`
| 64-bit integer (signed) | `torch.int64` or `torch.long` | `torch.*.LongTensor`

