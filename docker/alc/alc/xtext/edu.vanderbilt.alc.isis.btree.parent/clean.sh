#!/bin/bash
sudo rm -rf build \
           .gradle/ \
           ./edu.vanderbilt.alc.isis.btree/build \
           ./edu.vanderbilt.alc.isis.btree/bin/* \
           ./edu.vanderbilt.alc.isis.btree.ide/build \
           ./edu.vanderbilt.alc.isis.btree.ide/bin/* \
           ./btree.vscode/build \
           ./btree.vscode/src/btree/bin/* \
           ./btree.vscode/src/btree/lib/* \
           ./btree.vscode/.gradle/ \
           ./btree.vscode/node_modules/ \
           ./btree.vscode/out/ \
           package-lock.json \
           ./btree.vscode/package-lock.json \
           node_modules
