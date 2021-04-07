#!/bin/bash
# ideas used from https://gist.github.com/motemen/8595451

# Based on https://github.com/eldarlabs/ghpages-deploy-script/blob/master/scripts/deploy-ghpages.sh
# Used with their MIT license https://github.com/eldarlabs/ghpages-deploy-script/blob/master/LICENSE

# abort the script if there is a non-zero error
set -e

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
