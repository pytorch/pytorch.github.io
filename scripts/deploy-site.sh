#!/bin/bash
# ideas used from https://gist.github.com/motemen/8595451

# Based on https://github.com/eldarlabs/ghpages-deploy-script/blob/master/scripts/deploy-ghpages.sh
# Used with their MIT license https://github.com/eldarlabs/ghpages-deploy-script/blob/master/LICENSE

# abort the script if there is a non-zero error
set -ex

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
git remote add origin "$remote"
git fetch --depth 1

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
make build_deploy
cd master-branch

# copy over or recompile the new site
cp -a "../_site/." .

# have small jekyll config to allow underscores
echo "include: [_static, _images, _modules, _sources, _asserts.html, _creation.html, _comparison.html, _lowrank.html, _script.html, _diagnostic.html, _dynamo.html, _serialization.html, _type_utils, _tensor_str.html, _trace.html, _utils.html, _internal, _C, _distributed_autograd.html, _distributed_c10d.html, _distributed_rpc.html, _fft.html, _linalg.html, _monitor.html, _nested.html, _nn.html, _profiler.html, _sparse.html, _special.html, __config__.html, _dynamo, _lobpcg.html, _jit_internal.html, _numeric_suite.html, _numeric_suite_fx.html, _sanitizer.html, _symbolic_trace.html, _async.html, _freeze.html, _fuser.html, _type_utils.html, _utils ]" > _config.yml

# stage any changes and new files
git add -A
# now commit, ignoring branch master doesn't seem to work, so trying skip
git commit --allow-empty -m "Deploy to GitHub Pages on master [ci skip]"
# and push, but send any output to /dev/null to hide anything sensitive
git push --force --quiet https://pytorchbot:$SECRET_PYTORCHBOT_TOKEN@github.com/pytorch/pytorch.github.io.git master
# go back to where we started and remove the master git repo we made and used
# for deployment
cd ..
rm -rf master-branch

echo "Finished Deployment!"
