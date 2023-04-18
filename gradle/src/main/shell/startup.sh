#!/bin/bash

DIRECTORY="$(dirname "$0")"

export ALC_HOME=$ALC_SRC
export ALC_WORKING_DIR=$ALC_WORK
export ALC_FILESHARE_CONFIG_FILE=$ALC_FILESHARE_CONFIG_FILE1
#source /opt/ros/kinetic/setup.bash
source /opt/ros/melodic/setup.bash

OUTPUT_DIRECTORY=/alc/workflows/workflow_loop
mkdir -p "$OUTPUT_DIRECTORY"

cp $ALC_HOME/alc_utils/docker/run_assurance_gen.sh /root/.
chmod +x /root/run_assurance_gen.sh
export PYTHONPATH="$ALC_HOME/alc_utils/LaunchActivity:/alc/webgme/automate/gradle/src/main/pythonep:/alc/webgme/src/coomon/python:/alc/assurancecasetools/:/alc/resonate/resonate:$PYTHONPATH"
python3.6 $DIRECTORY/../python/workflow_loop.py > "$OUTPUT_DIRECTORY/stdout" 2> "$OUTPUT_DIRECTORY/stderr" &
npm run users -- useradd -c -s alc admin@mail.com alc
cp $ALC_HOME/webgme/ALC-Dev/package.json /alc/webgme/.

# npm install  -g @supercharge/strings
# npm install -g jsonrpc-ws-proxy js-yaml
# SERVER_XML="$(curl -sSL "https://pvsc.blob.core.windows.net/python-language-server-stable?restype=container&comp=list&prefix=Python-Language-Server-linux-x64")"
# NUPKG_LINK="$(echo "$SERVER_XML" | grep -Eo 'https://[^<]+\.nupkg' | tail -n1)"
# wget -nv "${NUPKG_LINK}"
# unzip -d /opt/mspyls Python-Language-Server-linux-x64.*.nupkg
# chmod +x /opt/mspyls/Microsoft.Python.LanguageServer
# ln -s /opt/mspyls/Microsoft.Python.LanguageServer /usr/local/bin/Microsoft.Python.LanguageServer
# rm Python-Language-Server-linux-x64.*.nupkg

#jsonrpc-ws-proxy --port 5000 --languageServers /alc/webgme/languageservers/languageServers.yml&
source $HOME/.nvm/nvm.sh
nvm use v8.16.2 
npm start&
