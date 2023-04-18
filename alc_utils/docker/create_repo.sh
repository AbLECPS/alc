#!/bin/bash

echo "git-repo initialization"
### create bluerov.git
export repo_name=$1
export src_folder=$3
export force=$2


echo $repo_name
echo $src_folder
echo $force

#if [[ $# -ne 2 ]]; then
#export src_folder=$2
#fi

export ALC_GITSERVER_ROOT=$ALC_WORKING_DIR/.gitserver
file=$ALC_GITSERVER_ROOT/repos/${repo_name}.git
echo "here"
export force_val=1
if [[ "$force" == "$force_val" ]]; then
   echo "deleting existing repo (if exists)"
   if [[ -e "$file" ]]; then
      rm -rf $file
   fi;
else
   echo "not deleting existing repo (if exists)"
   
fi;

echo "here1"
if [[ ! -e "$file" ]]; then
  mkdir -p $ALC_GITSERVER_ROOT/repos/temp$repo_name
  pushd $ALC_GITSERVER_ROOT/repos/temp$repo_name
  cp -r $ALC_HOME/alc_utils .
  cp -r $ALC_HOME/alc_ros .
  find .

  if [[ $repo_name == "alc_core" ]]; then
    cp -r $ALC_HOME/alc_utils/alc_core/model .
    find .
  fi

  if [[ -z "$src_folder" ]]; then
        echo "no user content for the repo"
  else
        cp -r $src_folder/*  .
        #rm -rf $src_folder
  fi

  build_file="build.sh"
  if [[ ! -e "$build_file" ]]; then
    cp $ALC_HOME/alc_utils/docker/repo_build_template.sh  ./build.sh
  fi 
  chmod +x ./build.sh

  
  activity_file="./model/activity_definitions/readme.md"
  if [[ ! -e "$activity_file" ]]; then
    mkdir -p ./model/activity_definitions
    echo "activities folder" >> ./model/activity_definitions/readme.md
  fi 
  
  find . -type d -name ".git" -exec rm -rf {} +
  git init --shared=true
  git add -A
  git config --local user.name "admin"
  git config user.email "admin@alc.alc"
  git commit -m "initial setup"
  popd

  pushd $ALC_GITSERVER_ROOT/repos
  git clone --bare temp$repo_name $repo_name.git
  rm -rf temp$repo_name
  chown -R 1000:1000 $repo_name.git
  popd
else
  echo "repo $repo_name exists in the repo already"
  exit 1
fi
exit 0
