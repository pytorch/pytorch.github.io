# shiftlab/pytorch

A static website, built in [Jekyll](https://jekyllrb.com/) and [Bootstrap](https://getbootstrap.com/) for [Pytorch]()https://pytorch.org/), and its tutorials and documentation.

## Getting Started
Read up a bit on [Jekyll](https://developer.chrome.com/apps/first_app) static sites.

Install Jekyll on your local system:

```
# Install Jekyll

$ gem install jekyll bundler
```

Read up on [Jekyll](https://developer.chrome.com/apps/first_app) static websites.

### Local Development

To run the website locally for development:

```
# Build the site on the preview server
make serve

# Now browse to http://localhost:4000
```

### Building the Static Site

To build the static website from source:

```
jekyll build --source _source --destination _deploy
```

### Deployments
The website is hosted on [Github Pages](https://pages.github.com/). To deploy changes, merge your latest code to `master` branch and run `$ cmd tbd`.
