set -ex
pushd _hub
git pull https://github.com/pytorch/hub
popd
cp _hub/images/* assets/images/

pip install notedown
mkdir -p assets/hub/

pushd _hub
find . -maxdepth 1 -name "*.md" | grep -v "README" | cut -f2- -d"/" |
    while read file; do
        tail -n +15 "$file" | notedown >"../assets/hub/${file%.md}.ipynb";
    done
popd
