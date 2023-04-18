from plugin_utils import PluginExtensionBase
import ac_construction as acc
from .model_interface import ModelInterface
from .utils import tuple_to_node


class AssuranceCaseConstructor(PluginExtensionBase):
    def __init__(self, *args, **kwargs):
        # Call base class init
        super(AssuranceCaseConstructor, self).__init__(*args, **kwargs)
        self.core = self.parent_plugin.core
        self.root_node = self.parent_plugin.root_node
        self.print_reason = False #print_reason

        # Find "ALC", "Modeling", and "V&V&A" top-level objects
        self.alc_root = self.find_child_by_meta_type(self.root_node, "ALC")
        self.modeling_root = self.find_child_by_meta_type(self.alc_root, "Modeling")
        self.vva_root = self.find_child_by_meta_type(self.alc_root, "V&V&A")
        self.assurance_root = self.find_child_by_meta_type(self.vva_root, "Assurance")

        # Initialize interface to WebGME models
        self.logger.info("Building interface to WebGME models...")
        self.models = ModelInterface(self, self.modeling_root)
        self.root_id = self.modeling_root["rootId"]
        self.logger.info("Done building WebGME model interface.")

        # Load pattern library
        self.pat_lib = acc.PatternLibrary()
        self.pat_lib.load_default_lib()
    

    def main(self):
        ac = self.gen_assurance_case()
        self.build_ac_in_webgme(ac)

    def find_child_by_meta_type(self, root, meta_type):
        """
        Function to find a child of specified root with the desired meta_type.
        Returns the first such node and does not look for any additional matches.

        :raises ValueError: if no child with the desired type exists
        """
        meta_node = self.get_meta(meta_type)
        for node in self.core.load_children(root):
            if self.core.is_instance_of(node, meta_node):
                return node
        raise ValueError("No node of type %s found as a child of node %s" % (meta_type, str(root)))

    def gen_assurance_case(self):
        # Initialize assurance case
        assurance_case = acc.AssuranceCase()

        # Develop the assurance case until all nodes have been addressed
        undeveloped_nodes = [None]
        while len(undeveloped_nodes) > 0:
            node = undeveloped_nodes.pop(0)

            # Determine valid pattern candidates
            if node is None:
                # If node is 'None', this is the initial pattern and all patterns in library are valid options
                pattern_opts = self.pat_lib.list_patterns()
            else:
                # Otherwise, suggest suitable pattern choices based on variable type
                var_types = [var.formal_base.type for var in node.parameters]
                if len(var_types) == 0:
                    # FIXME: What if node does not have a variable? Any other way to develop it?
                    continue
                elif len(var_types) == 1:
                    var_type = var_types[0]
                else:
                    # FIXME: This probably needs to support multiple variables
                    raise ValueError("Too many variables")
                pattern_opts = self.pat_lib.find_by_root_variable_type(var_type)

            # Prompt user for desired pattern
            if len(pattern_opts) > 0:
                selections = acc.select_option(pattern_opts, prompt_text="Please select a pattern for instantiation:")
                selected_pattern = pattern_opts[selections[0]].instantiate()#self.print_reason)
            else:
                self.logger.warn("No patterns available for further development of node %s. Skipping." % str(node))
                continue

            # Instantiate selected pattern with undeveloped node as a base
            selected_pattern.instantiate(self.models, base_node=node, print_reason = self.print_reason)
            selected_pattern.parameter_structure.print_parameter_assignments()
            selected_pattern.node_structure.print_argument_nodes()

            # If pattern was instantiated successfully, add to assurance case
            if len(selected_pattern.node_structure.argument.nodes) > 0:
                assurance_case.add_pattern(selected_pattern, node)

            # Add any remaining undeveloped nodes to list
            undeveloped_nodes.extend(selected_pattern.find_undeveloped_nodes())

        return assurance_case

    def build_ac_in_webgme(self, assurance_case):
        """Builds WebGME model which corresponds to networkx representation of assurance case."""
        # Prompt user for name of new assurance case model
        ac_name = input("Enter a name for the assurance case model:")

        # Create initial GSN model for this assurance case
        gsn_base = self.get_meta("SEAM.GSN_Model")
        ac_root = self.core.create_child(self.assurance_root, gsn_base)
        self.core.set_attribute(ac_root, "name", ac_name)

        # For each instantiated pattern in the assurance case, construct a GSN model
        graph_to_webgme_node_map = {}
        graph_to_webgme_parameter_map = {}
        for pattern, parent_node in assurance_case.pattern_info:
            # Create empty GSN model
            gsn_model = self.core.create_child(ac_root, gsn_base)
            self.core.set_attribute(gsn_model, "name", pattern.name)

            # Create an object in the GSN model for each node in the graph
            pos = acc.utils.tree_layout(pattern.node_structure.argument, scale=300, y_dir=1, centered=False)
            for node in pattern.node_structure.argument.nodes:
                # Create an empty object of correct type
                meta_type_name = "SEAM.%s" % node.outline.type
                meta_base = self.get_meta(meta_type_name)
                webgme_node = self.core.create_child(gsn_model, meta_base)

                # Initialize values in new object
                attributes = {"name": str(node), "description": node.desc, "In Development": node.undeveloped}
                for key, value in attributes.items():
                    self.core.set_attribute(webgme_node, key, value)

                # Set position of node appropriately
                node_pos = {"x": int(pos[node][0]), "y": int(pos[node][1])}
                self.core.set_registry(webgme_node, "position", node_pos)

                # Maintain map from networkx graph nodes to WebGME nodes
                graph_to_webgme_node_map[node] = webgme_node

                # For any parameters in this node, create an appropriate reference to the source model
                for param in node.parameters:
                    # Create reference object and set attributes
                    meta_base = self.get_meta("SEAM.ModelRef")
                    model_ref = self.core.create_child(webgme_node, meta_base)

                    element_type = param.formal_base.type[0]
                    #if (param.type):
                    #    element_type = param.type
                    attributes = {"name": str(param), "element_type": str(element_type)}
                    for key, value in attributes.items():
                        self.core.set_attribute(model_ref, key, value)

                    # Set pointer to source model object.
                    # `param.value` is a tuple representing a WebGME node. Translate to dictionary then set pointer.
                    source_model_node = tuple_to_node(param.value, self.root_id)
                    self.core.set_pointer(model_ref, "Ref", source_model_node)

                    # FIXME: Same parameter can be used in multiple GSN nodes. This should be a 1-to-many mapping
                    # Maintain map from networkx parameter nodes to WebGME nodes
                    graph_to_webgme_parameter_map[param] = model_ref

            # Create a connector in the GSN model for each edge in the graph
            for edge in pattern.node_structure.argument.edges(data="object"):
                src_node, dst_node, edge_obj = edge

                # Create an empty connector of correct type
                meta_type_name = "SEAM.%s" % edge_obj.type
                meta_base = self.get_meta(meta_type_name)
                webgme_edge = self.core.create_child(gsn_model, meta_base)

                # Set connector `src` and `dst`
                self.core.set_pointer(webgme_edge, "src", graph_to_webgme_node_map[src_node])
                self.core.set_pointer(webgme_edge, "dst", graph_to_webgme_node_map[dst_node])

            # If this pattern has a parent, create an appropriate `SupportRef` link in the parent model
            if parent_node is not None:
                # Find parent node and determine which GSN model object it belongs to
                webgme_parent = graph_to_webgme_node_map[parent_node]
                parent_gsn = self.core.get_parent(webgme_parent)

                # Create `SupportRef` link pointing to root of current pattern
                meta_base = self.get_meta("SEAM.SupportRef")
                support_ref = self.core.create_child(parent_gsn, meta_base)
                pattern_root = pattern.node_structure.argument.find_root_nodes()[0]
                self.core.set_pointer(support_ref, "Ref", graph_to_webgme_node_map[pattern_root])

                # Make connection from parent node to SupportRef
                meta_base = self.get_meta("SEAM.SupportedBy")
                support_ref_conn = self.core.create_child(parent_gsn, meta_base)
                self.core.set_pointer(support_ref_conn, "src", webgme_parent)
                self.core.set_pointer(support_ref_conn, "dst", support_ref)

            # For all parameters in the pattern, add `descendant` references to all child parameters to maintain explainability
            for parameter in pattern.parameter_structure.graph.nodes():
                model_ref = graph_to_webgme_parameter_map[parameter]
                for i, child_param in enumerate(pattern.parameter_structure.graph.successors(parameter)):
                    # Create `Descendant` reference object and set attributes
                    descendant = self.core.create_child(model_ref, self.get_meta("SEAM.Descendant"))
                    attributes = {"name": str(child_param), "Explanation": str(child_param.path)}
                    for key, value in attributes.items():
                        self.core.set_attribute(descendant, key, value)
                    self.core.set_registry(descendant, "position", {"x": 100 + 50 * i, "y": 100 + 50 * i})

                    # Set pointer to child parameter
                    # FIXME: Shouldn't need to set pointer meta target once meta is updated
                    # FIXME: Same parameter can be used in multiple GSN nodes. This should be a set relationship, not pointer
                    self.core.set_pointer_meta_target(descendant, "child", graph_to_webgme_parameter_map[child_param])
                    self.core.set_pointer(descendant, "child", graph_to_webgme_parameter_map[child_param])
