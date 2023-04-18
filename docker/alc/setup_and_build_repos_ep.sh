#!/bin/bash
set -e


file=~/.ssh
if [[ ! -e "$file" ]]; then
    mkdir ~/.ssh
    chmod 755 ~/.ssh
    cp -r $ALC_DOCKERFILES/sshcontents/* ~/.ssh/.
    chmod 600 ~/.ssh/gitkey
    cat $ALC_DOCKERFILES/keys/gitkey.pub >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
fi


PYTHONPATHOLD=$PYTHONPATH
export PYTHONPATH=$ALC_HOME:$PYTHONPATHOLD

if ([[ $1 == "ALL" ]] || [[ $1 == "alc_core" ]]);
then 
pushd $ALC_HOME/alc_utils
echo "setting up alc_core repo"
python setup_alc_core.py
echo "done setting up alc_core repo"
echo "build alc_core repo"
python build_alc_core.py
echo "done build alc_core repo"
popd
fi;

if ([[ $1 == "ALL" ]] || [[ $1 == "bluerov2_standalone" ]]);
then
pushd $ALC_HOME/alc_utils
echo "setting up bluerov2standalone repo"
python setup_bluerov2_standalone.py
echo "done setting up bluerov2standalone repo"
echo "build bluerov2standalone repo"
python build_bluerov2_standalone.py
echo "done build bluerov2standalone repo"
popd
fi;


