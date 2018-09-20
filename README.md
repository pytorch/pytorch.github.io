# pytorch.org site

[https://pytorch.org](https://pytorch.org)

A static website built in [Jekyll](https://jekyllrb.com/) and [Bootstrap](https://getbootstrap.com/) for [PyTorch](https://pytorch.org/), and its tutorials and documentation.

## Prerequisites

Install the following packages before attempting to setup the project:

- [rbenv](https://github.com/rbenv/rbenv)
- [ruby-build](https://github.com/rbenv/ruby-build)
- [nvm](https://github.com/creationix/nvm)

## Setup

#### Install required Ruby version:

```
#### You only need to run these commands if you are missing the needed Ruby version.

rbenv install `cat .ruby-version`
gem install bundler
rbenv rehash

####

bundle install
rbenv rehash
```

#### Install required Node version

```
nvm install
nvm use
```

#### Install Yarn

```
brew install yarn --without-node
yarn install
```

## Local Development

To run the website locally for development:

```
make serve
```

Then navigate to [localhost:4000](localhost:4000).

Note the `serve` task is contained in a `Makefile` in the root directory. We are using `make` as an alternative to the standard `jekyll serve` as we want to run `yarn`, which is not included in Jekyll by default.

### Building the Static Site

To build the static website from source:

```
make build
```

This will build the static site at `./_site`. This directory is not tracked in git.

## Deployments

The website is hosted on [Github Pages](https://pages.github.com/) at [https://pytorch.org](https://pytorch.org).

To deploy changes, merge your latest code into the `sites` branch. A build will be automatically built and committed to the `master` branch via a CircleCI job.

To view the status of the build visit [https://circleci.com/gh/pytorch/pytorch.github.io](https://circleci.com/gh/pytorch/pytorch.github.io).
