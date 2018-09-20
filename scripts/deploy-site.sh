#!/bin/sh
# ideas used from https://gist.github.com/motemen/8595451

# Based on https://github.com/eldarlabs/ghpages-deploy-script/blob/master/scripts/deploy-ghpages.sh
# Used with their MIT license https://github.com/eldarlabs/ghpages-deploy-script/blob/master/LICENSE

# abort the script if there is a non-zero error
set -e

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
make build_deploy
cd master-branch

# copy over or recompile the new site
cp -a "../_site/." .

# have small jekyll config to allow underscores
echo "include: [_static, _images, _modules, _sources, _tensor_str.html, _utils.html]" > _config.yml

# stage any changes and new files
git add -A
# now commit, ignoring branch master doesn't seem to work, so trying skip
git commit --allow-empty -m "Deploy to GitHub Pages on master [ci skip]"
# and push, but send any output to /dev/null to hide anything sensitive
git push --force --quiet https://facebook-circleci-bot:$CIRCLECI_PUBLISH_TOKEN@github.com/pytorch/pytorch.github.io.git master
# go back to where we started and remove the master git repo we made and used
# for deployment
cd ..
rm -rf master-branch

echo "Finished Deployment!"
