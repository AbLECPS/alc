#!/bin/bash
# Adapted from https://github.com/giovtorres/slurm-docker-cluster/blob/master/register_cluster.sh
docker exec alc_alc bash -c "/usr/bin/sacctmgr --immediate add cluster name=cluster" && \
docker-compose restart alc_slurmdbd alc_alc