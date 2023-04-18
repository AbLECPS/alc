#!/bin/bash

# Exit on error, don't suppress
set -e


# Execute job
pushd $ALC_WORKING_DIR/{{ relative_result_dir|string }}
python main.py
popd
