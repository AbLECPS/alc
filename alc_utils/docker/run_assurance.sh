#!/bin/bash
#run the assurance case generator
args="$@" 
ssh -i ~/.ssh/gitkey root@$ALC_SSH_HOST -t "./run_assurance_gen.sh $args"
