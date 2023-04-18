#!/bin/bash

pushd /alc/webgme/src/plugins/AssuranceCaseConstruction
export PYTHONPATH=$PYTHONPATH:/alc/webgme/src/common/python
python3.6 run_assurance.py $@
popd
