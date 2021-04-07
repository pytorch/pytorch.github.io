#!/bin/bash

set -e

# initialize hub submodule
git submodule deinit -f . && git submodule update --init --recursive

# use latest hub
./_devel/update_hub_submodule.sh

# Files not related to build should be deleted.
pushd _hub
rm -R `ls -1 -d */`
rm -f README.md
popd

# show where we are on the machine
pwd
remote=$(git config remote.origin.url)

# make a directory to put the master branch
mkdir master-branch
cd master-branch
# now lets setup a new repo so we can update the master branch
git init
git remote add --fetch origin "$remote"

# switch into the the master branch
if git rev-parse --verify origin/master > /dev/null 2>&1
then
    git checkout master
    # delete any old site as we are going to replace it
    # Note: this explodes if there aren't any, so moving it here for now
    git rm -rf .
else
    git checkout --orphan master
fi

cd "../"
(
  set -x
  make build_deploy
)
cd master-branch

# copy over or recompile the new site
cp -a "../_site/." .

# have small jekyll config to allow underscores
echo "include: [_static, _images, _modules, _sources, _tensor_str.html, _utils.html]" > _config.yml
