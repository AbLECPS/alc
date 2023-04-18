import math
import sys
from six.moves import input
import numpy as np


def select_option(options,
                  multiplicity=(1, 1),
                  desc_text="Options:",
                  prompt_text="Please select from the above options (comma-separated for multiple selection,"
                              " 'ALL' to select all options, or 'MORE' to see additional options.):",
                  addl_opts_avail=False):
    # Validate and unpack arguments
    if not type(multiplicity) == tuple:
        raise TypeError("Expected multiplicity to be of type `Tuple(Int, Int)`, but got type '%s'" % type(multiplicity))
    lower_bound, upper_bound = multiplicity
    if type(lower_bound) != int or type(upper_bound) != int:
        raise TypeError("Provided multiplicity argument contained non-integer value.")

    # Print out description and options
    print(desc_text)
    for i, option in enumerate(options):
        print("\t(%d) %s" % (i, str(option)))

    while True:
        # Wait for user input. If input contains commas, interpret as list.
        # Otherwise interpret as integer first, then as string.
        in_text = input(prompt_text).replace(" ", "")
        if "," in in_text:
            selections = [int(item) for item in in_text.split(",")]
        else:
            try:
                selections = [int(in_text)]
            except ValueError:
                if in_text.lower() == "all":
                    selections = range(len(options))
                elif in_text.lower() == "more" and addl_opts_avail:
                    return -1
                else:
                    print("Invalid input.")
                    continue

        # Validate multiplicity of selected options
        if lower_bound == upper_bound and len(selections) != lower_bound:
            print("Must select exactly %d items." % lower_bound)
            continue
        elif not (lower_bound <= len(selections) <= upper_bound):
            if upper_bound >= sys.maxsize:
                print("Must select at least %s items." % str(lower_bound))
            else:
                print("Must select between %s and %s items." % (str(lower_bound), str(upper_bound)))
            continue

        # Validate selected items were on the list
        invalid_selection = False
        for selection in selections:
            if not (0 <= selection < len(options)):
                print("Invalid selection '%d'." % selection)
                invalid_selection = True
        if invalid_selection:
            continue

        return selections


def tree_layout(g, scale=1, x_dir=1, y_dir=-1, centered=True):
    """Layout function for displaying Trees (based on NetworkX graphs).
    For each node, determines X and Y position based on node depth and number of other nodes at the same depth."""
    pos = {}
    next_level_nodes = g.find_root_nodes()
    while len(next_level_nodes) > 0:
        nodes = next_level_nodes
        next_level_nodes = []
        for i, node in enumerate(nodes):
            y = g.depth(node) * y_dir * scale
            if centered:
                x = (i + 1 - math.floor(len(nodes) / 2)) * x_dir * scale
            else:
                x = (i + 1) * x_dir * scale
            pos[node] = np.asarray([x, y])
            next_level_nodes.extend(list(g.successors(node)))
    return pos
