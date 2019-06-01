pushd _hub
git pull https://github.com/pytorch/hub
popd
cp _hub/images/* assets/images/
