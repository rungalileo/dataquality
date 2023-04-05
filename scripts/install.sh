#!/bin/sh -ex

pip install --upgrade pip
pip install flit

flit install --deps=develop --symlink
