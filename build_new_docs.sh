#!/usr/bin/env bash

# inputs:
# $1 = desired torch version i.e. (0.4.1)
# $2 = old torch version i.e. (0.4.0)

set -ex

result=${PWD##*/}
if [ -n result "pytorch.github.io" ]
then
  echo "This script must be run from pytorch.github.io/"
  exit(1)
end

# XXX: assumes "python" executable
torch_version="$(python -c "import torch; print(torch.__version__)")"
if [ -n torch_version $1 ]
then 
  echo "Specified $1 docs but found ${torch_version} build in environment."
  exit(1)
end

# Get all the documentation sources, put them in one place

# Clones pytorch, but it is assumed you have a binary installed already.
git clone https://github.com/pytorch/pytorch
# Checkout release branch
git checkout v$1
pushd pytorch

# Clone & build torchvision
git clone https://github.com/pytorch/vision
pushd vision
conda install -y pillow
time python setup.py install
popd
pushd docs
rm -rf source/torchvision
cp -r ../vision/docs/source source/torchvision

# Build the docs
pip install -r requirements.txt || true
make html-stable  # TODO: need to overwrite the version number with the new one...

popd # docs
popd # pytorch

cp pytorch/docs/build/html docs/release
cd docs

# sed to add the version selector
find docs/release -name "*.html" -print0 | xargs -0 sed -i 's/master ($1 )/$1 <br\/> <a href="http:\/\/pytorch.org\/docs\/versions.html"> version selector \&#x25BC<\/a>/g'

# Perform the shuffles
# Move the redirects folder
git mv $2 $1
# Archive stable
git mv stable $2
# Create a new stable based off of the latest build 
mv release stable

# NOTE:
# The following things need to be done manually.

# Fix redirects
# the $1/ folder must have a redirect file for every page
# under the new stable/ folder.
echo "Be sure to check that the $1/ redirects to stable/ are comprehensive"

# Change the version selector.
echo "Be sure to update the version selector"

