"""
This is where the implementation of the plugin code goes.
The Resonate-class is imported from both run_plugin.py and run_debug.py
"""
import sys
import logging
from webgme_bindings import PluginBase
from resonate import general_analyze

# Setup a logger
logger = logging.getLogger('Resonate')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # By default it logs to stderr..
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class Resonate(PluginBase):
    def __init__(self, *args, **kwargs):
        # Call base class init
        super(Resonate, self).__init__(*args, **kwargs)

        # Support both "" and "ALCMeta" namespaces.
        if self.namespace == "":
            self.namespace_prefix = "ALCMeta."
        elif self.namespace == "ALCMeta":
            self.namespace_prefix = ""
        else:
            raise ValueError("Unsupported namespace (%s)." % self.namespace)

    def main(self):
        core = self.core
        root_node = self.root_node
        active_node = self.active_node

        name = core.get_attribute(active_node, 'name')

        # Load plugin arguments
        config_dict = self.get_current_config()
        required_path = config_dict["execution_path"]
        if required_path is not None:
            artifact_content = general_analyze.run_general_analyze_script(config_dict["execute_recent"], required_path)
        else:
            artifact_content = general_analyze.run_general_analyze_script(config_dict["execute_recent"])

        logger.info('ActiveNode at "{0}" has name {1}'.format(core.get_path(active_node), name))

        core.set_attribute(active_node, 'name', 'newName')

        commit_info = self.util.save(root_node, self.commit_hash, 'master', 'Python plugin updated the model')
        logger.info('committed :{0}'.format(commit_info))

        return self.add_artifact("Analysis Outcomes", artifact_content)

