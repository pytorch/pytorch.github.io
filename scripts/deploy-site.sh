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
git config --global user.email "facebook-circleci-bot@users.noreply.github.com" > /dev/null 2>&1
git config --global user.name "Website Deployment Script" > /dev/null 2>&1
echo "machine github.com login facebook-circleci-bot password $CIRCLECI_PUBLISH_TOKEN" > ~/.netrc
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

# stage any changes and new files
git add -A
# now commit, ignoring branch master doesn't seem to work, so trying skip
git commit --allow-empty -m "Deploy to GitHub Pages on master [ci skip]"
# and push, but send any output to /dev/null to hide anything sensitive
git push --force --quiet origin master
# go back to where we started and remove the master git repo we made and used
# for deployment
cd ..
rm -rf master-branch

echo "Finished Deployment!"
