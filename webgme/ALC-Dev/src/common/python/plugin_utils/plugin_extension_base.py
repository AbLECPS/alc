# PluginExtensionBase provides a core set of functionality for writing a Python "PluginExtension".
# This concept is not defined by WebGME itself, but allows python code to be shared/reused between plugins.
# Organizing code this way is a workaround since Python-based plugins currently CANNOT invoke other plugins.


class PluginExtensionBase(object):
    def __init__(self, parent_plugin):
        # Store parent plugin
        self.parent_plugin = parent_plugin
        self.logger = self.parent_plugin.logger

        # Support both "" and "ALCMeta" namespaces.
        if self.parent_plugin.namespace == "":
            self.namespace_prefix = "ALC_EP_Meta."
            if not self.parent_plugin.META.get(self.namespace_prefix + "AssemblyModel", None):
                self.namespace_prefix = "ALCMeta."
        elif self.parent_plugin.namespace == "ALC_EP_Meta" or self.parent_plugin.namespace == "ALCMeta":
            self.namespace_prefix = ""
        else:
            self.raise_and_notify("Unsupported namespace (%s)." % self.parent_plugin.namespace, ValueError)

    # Wrapper function to automatically use correct META prefix
    # eg. "AssemblyModel" may be "ALCMeta.AssemblyModel" or just "AssemblyModel"
    def get_meta(self, meta_type):
        return self.parent_plugin.META[self.namespace_prefix + meta_type]

    # Alternative way to raise exceptions. This function sends a notification back to the WebGME user before raising
    # the specified exception. Otherwise, users have no way to know why the plugin failed
    def raise_and_notify(self, err_msg, exception_type):
        msg = {"message": err_msg,
               "severity": "error"}
        self.parent_plugin.send_notification(msg)
        raise exception_type(err_msg)