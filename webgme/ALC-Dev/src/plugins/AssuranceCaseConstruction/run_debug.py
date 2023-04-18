"""
This file can be used as the entry point when debugging the python portion of the plugin.
Rather than relying on be called from a node-process with a corezmq server already up and running
(which is the case for run_plugin.py) this script starts such a server in a sub-process.

To change the context (project-name etc.) modify the CAPITALIZED options passed to the spawned node-js server.

Note! This must run with the root of the webgme-repository as cwd.
"""

import sys
import os
import subprocess
import signal
import atexit
import logging
from webgme_bindings import WebGME
from AssuranceCaseConstruction import AssuranceCaseConstruction

os.environ['NODE_ENV'] = 'assurancecase'

if (len(sys.argv) < 2 ):
    usage_str ='''
usage: run_assurance <PROJECT_NAME> [-r ]
option:
   -r : prints reason
'''
    print (usage_str)
    exit(1)




logger = logging.getLogger('AssuranceCaseConstruction')

# Modify these or add option or parse from sys.argv (as in done in run_plugin.py)
PORT = '5555'
#PROJECT_NAME = 'bl1'
PROJECT_NAME = sys.argv[1]
BRANCH_NAME = 'master'
ACTIVE_NODE_PATH = ''
ACTIVE_SELECTION_PATHS = []
NAMESPACE = ''
METADATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'metadata.json')

COREZMQ_SERVER_FILE = os.path.join('/alc/webgme', 'node_modules', 'webgme-bindings', 'bin', 'corezmq_server.js')

if not os.path.isfile(COREZMQ_SERVER_FILE):
    COREZMQ_SERVER_FILE = os.path.join(os.getcwd(), 'bin', 'corezmq_server.js')

# Star the server (see bin/corezmq_server.js for more options e.g. for how to pass a pluginConfig)
node_process = subprocess.Popen(['node', COREZMQ_SERVER_FILE, PROJECT_NAME, '-p', PORT, '-m', METADATA_PATH, '-u', 'admin:vanderbilt'],
                                stdout=sys.stdout, stderr=sys.stderr)

logger.info('Node-process running at PID {0}'.format(node_process.pid))
# Create an instance of WebGME and the plugin
webgme = WebGME(PORT, logger)


def exit_handler():
    logger.info('Cleaning up!')
    webgme.disconnect()
    node_process.send_signal(signal.SIGTERM)


atexit.register(exit_handler)

commit_hash = webgme.project.get_branch_hash(BRANCH_NAME)
print_reason = False
if (len(sys.argv)==3):
    if (sys.arv[2]=='-r'):
        print_reason = True
#kwargs = {'print_reason': print_reason}
plugin = AssuranceCaseConstruction(webgme, commit_hash, BRANCH_NAME, ACTIVE_NODE_PATH, ACTIVE_SELECTION_PATHS, NAMESPACE)

# Do the work
plugin.main()

# The exit_handler will be invoked after this line
