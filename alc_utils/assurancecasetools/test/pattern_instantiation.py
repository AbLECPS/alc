from ac_construction.pattern_library import PatternLibrary
from ac_construction.assurance_case import AssuranceCase
from ac_construction.utils import select_option
from ac_construction.models import example_model_set


if __name__ == "__main__":
    # Load Pattern Library
    lib = PatternLibrary()
    lib.load_default_lib()

    # Start with Risk Reduction Pattern
    rrp = lib.find_by_name("risk_reduction").instantiate()
    rrp.draw_structure_graphs()

    # Instantiate pattern, print results, and draw instance graphs
    rrp.instantiate(example_model_set)
    rrp.parameter_structure.print_parameter_assignments()
    rrp.node_structure.print_argument_nodes()
    rrp.draw_instance_graphs()

    # Initialize assurance case
    ac = AssuranceCase()
    ac.add_pattern(rrp)

    # Find undeveloped nodes in the pattern and try to further develop them
    for node in rrp.find_undeveloped_nodes():
        var_types = [var.formal_base.name for var in node.parameters]
        if len(var_types) == 0:
            # FIXME: What if node does not have a variable? Any other way to develop it?
            continue
        elif len(var_types) == 1:
            var_type = var_types[0]
        else:
            # FIXME: This probably needs to support multiple variables
            raise ValueError("Too many variables")

        # Find patterns with matching variable type
        pattern_opts = lib.find_by_root_variable_type(var_type)
        selections = select_option(pattern_opts)
        selected_pattern = pattern_opts[selections[0]].instantiate()

        # Instantiate selected pattern with underdeveloped node as a base
        selected_pattern.instantiate(example_model_set, base_node=node)
        selected_pattern.parameter_structure.print_parameter_assignments()
        selected_pattern.node_structure.print_argument_nodes()

        # If pattern was instantiated successfully, add to assurance case
        if len(selected_pattern.node_structure.argument.nodes) > 0:
            ac.add_pattern(selected_pattern, node)

    # Write generated graphs to GraphML
    # rrp.node_structure.argument.serialize_edges()
    # nx.write_graphml(rrp.node_structure.argument, "generated_graphs/risk_reduction.graphml")
    # resonate_pattern.node_structure.argument.serialize_edges()
    # nx.write_graphml(resonate_pattern.node_structure.argument, "generated_graphs/resonate.graphml")

    # Draw generated assurance case
    ac.draw()

