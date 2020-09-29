set -ex
pushd _hub
git pull https://github.com/pytorch/hub
popd
cp _hub/images/* assets/images/

python3.8 -c 'import notedown' || pip install notedown
python3.8 -c 'import yaml' || pip install pyyaml
mkdir -p assets/hub/

pushd _hub
find . -maxdepth 1 -name "*.md" | grep -v "README" | cut -f2- -d"/" |
    while read file; do
        cat "$file" | python3.8 ../_devel/formatter.py | notedown >"../assets/hub/${file%.md}.ipynb";
    done
popd
