#!/bin/bash

FILE=$PWD/src/bluerov2/bluerov2_description/meshes/bluerov2_propcw.dae
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else 
    echo "$FILE does not exist."
    mkdir test_repo
    pushd test_repo
    git init
    git remote add -f origin https://github.com/fredvaz/bluerov2.git
    git config core.sparseCheckout true
    echo "bluerov2_description/meshes" >> .git/info/sparse-checkout
    git pull origin master
    popd
    mv $PWD/test_repo/bluerov2_description/meshes/* $PWD/src/bluerov2/bluerov2_description/meshes/
    rm -Rf $PWD/test_repo
fi