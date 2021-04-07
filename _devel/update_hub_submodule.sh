set -ex
pushd _hub
git pull https://github.com/pytorch/hub
popd
cp _hub/images/* assets/images/

python3 -c 'import notedown' || pip3 install notedown
python3 -c 'import yaml' || pip3 install pyyaml
mkdir -p assets/hub/

pushd _hub
find . -maxdepth 1 -name "*.md" | grep -v "README" | cut -f2- -d"/" |
    while read file; do
        cat "$file" | python3 ../_devel/formatter.py | notedown >"../assets/hub/${file%.md}.ipynb";
    done
popd
