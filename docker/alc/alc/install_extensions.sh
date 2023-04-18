#!/bin/bash
. /etc/profile.d/gradle.sh                                           
pushd /xtext/edu.vanderbilt.alc.isis.btree.parent                    
rm -rf build .gradle package-lock.json                               
rm -rf edu.vanderbilt.alc.isis.btree/build                           
rm -rf edu.vanderbilt.alc.isis.btree.ide/build                       
pushd btree.vscode                                                    
rm -rf build .gradle node_modules out package-lock.json              
popd                                                                 
./gradlew clean installDist vscodeExtension
pwd
ls btree.vscode
echo "*******************"
ls btree.vscode/build/
echo "*******************"
ls btree.vscode/build/vscode
echo "*******************"
cp btree.vscode/build/vscode/*.vsix /extension/.                                     
code-server --install-extension /extension/btree.vscode-unspecified.vsix        
popd
rm -rf /xtext/*
