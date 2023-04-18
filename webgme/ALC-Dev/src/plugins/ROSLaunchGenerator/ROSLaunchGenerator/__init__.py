"""
This is where the implementation of the plugin code goes.
The ROSLaunchGenerator-class is imported from both run_plugin.py and run_debug.py
"""
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>

from webgme_bindings import PluginBase
from ros_gen import SystemLaunchGen


class ROSLaunchGenerator(PluginBase):
    def __init__(self, *args, **kwargs):
        # Call base class init
        super(ROSLaunchGenerator, self).__init__(*args, **kwargs)

        # Initialize SystemLaunchGen extension
        self.sys_launch_gen = SystemLaunchGen(self)

    # Main function invoked when plugin is run, after class constructor has completed
    def main(self):
        artifact_content, _ = self.sys_launch_gen.gen_launch_file(self.active_node)
        self.result_set_success(True)
        return self.add_artifact("generated_launch_files", artifact_content)

