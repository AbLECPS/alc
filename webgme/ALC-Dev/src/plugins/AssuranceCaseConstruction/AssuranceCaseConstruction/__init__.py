"""
This is where the implementation of the plugin code goes.
The AssuranceCaseConstruction-class is imported from both run_plugin.py and run_debug.py
"""
import sys
import logging
from webgme_bindings import PluginBase
from assurance_case_constructor import AssuranceCaseConstructor

# Setup a logger
logger = logging.getLogger('AssuranceCaseConstruction')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # By default it logs to stderr..
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class AssuranceCaseConstruction(PluginBase):
    #def __init__(self, webgme, commit_hash, BRANCH_NAME, ACTIVE_NODE_PATH, ACTIVE_SELECTION_PATHS, NAMESPACE,
    #   config=None, **kwargs):
    def __init__(self, *args, **kwargs):
            
        # Call base class init
        super(AssuranceCaseConstruction, self).__init__(*args, **kwargs)
        #PluginBase.__init__(self, webgme, commit_hash, BRANCH_NAME, ACTIVE_NODE_PATH, ACTIVE_SELECTION_PATHS, NAMESPACE)
        #_print_reason = kwargs.get('print_reason', False)
               

        # Initialize SystemLaunchGen extension
        self.ac_constructor = AssuranceCaseConstructor(self)

    def main(self):
        self.ac_constructor.main()

        # Commit results
        commit_info = self.util.save(self.root_node, self.commit_hash, self.branch_name, 'Generated assurance case.')
        #self.result_set_success(True)
