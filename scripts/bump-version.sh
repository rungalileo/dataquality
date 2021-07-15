#!/bin/sh -e

current_version=$(python -c "import galileo_python; print(galileo_python.__version__)")
echo "current version: $current_version"
read -p "new version: " new_version

sed -i '' -e "s/$current_version/$new_version/g" galileo_python/__init__.py
