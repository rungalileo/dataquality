#!/bin/sh -e

current_version=$(python -c "import dataquality; print(dataquality.__version__)")
echo "current version: $current_version"
read -p "new version: " new_version

sed -i '' -e "s/$current_version/$new_version/g" dataquality/__init__.py
