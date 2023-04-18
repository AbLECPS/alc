#!/bin/bash
set -e

# Check that all necessary environment variables are set
if [[ -z "$NFS_SERVER_ADDR" ]]; then
  echo "NFS_SERVER_ADDR is not defined. Exiting."
  exit 0
fi
#if [[ -z "$NFS_SERVER_DIR" ]]; then
#  echo "NFS_SERVER_DIR is not defined. Exiting."
#  exit 0
#fi
if [[ -z "$NFS_LOCAL_DIR" ]]; then
  echo "NFS_LOCAL_DIR is not defined. Exiting."
  exit 0
fi
if [[ -z "$NFS_SERVER_PORT" ]]; then
  echo "Using default NFS port 2049."
  export NFS_SERVER_PORT=2049
fi

# Wait for NFS Server
until 2>/dev/null >/dev/tcp/$NFS_SERVER_ADDR/$NFS_SERVER_PORT; do
  echo "-- NFS Server is not available. Sleeping ..."
  sleep 1
done
echo "-- NFS Server is now active ..."

# Setup required NFS utilities and mount directory
mkdir -p "$NFS_LOCAL_DIR"
rpcbind
rpc.statd -d

# Mount NFS directory
echo "Attempting to mount with: 'mount -t nfs4 -o port=$NFS_SERVER_PORT $NFS_SERVER_ADDR:/ $NFS_LOCAL_DIR'"
mount -vvv -t nfs4 -o port=$NFS_SERVER_PORT $NFS_SERVER_ADDR:/ $NFS_LOCAL_DIR
echo "NFS mount finished."

# Execute command if provided
if [[ $# -eq 0 ]]; then
  exit 0
else
  exec "$@"
fi

