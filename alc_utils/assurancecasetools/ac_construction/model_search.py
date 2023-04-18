import sys


def find_paths(model_set, start_node, end_node, **kwargs):
    return _find_paths_bfs(model_set, start_node, end_node, **kwargs)


def _find_paths_dfs(model_set, start_node, end_node, path, visited):
    """Find all valid paths between `start_node` and `end_node` in the provided model set graph.
    Function recursively explores paths through the graph in a depth-first manner."""
    # If start_node and end_node are the same, this path is complete
    if start_node == end_node:
        return [path]

    # Initialize new paths list
    paths = []

    # First explore all outgoing edges from start_node
    for neighbor in model_set.successors(start_node):
        if neighbor not in visited:
            relation_type = model_set.edges[start_node, neighbor]["type"]
            step = [(start_node, neighbor, relation_type)]
            new_paths = _find_paths_dfs(model_set, neighbor, end_node, path + step, visited + [start_node])
            paths.extend(new_paths)

    # Then explore all incoming edges from start_node (i.e. opposite of edge direction)
    for neighbor in model_set.predecessors(start_node):
        if neighbor not in visited:
            relation_type = model_set.edges[neighbor, start_node]["type"]
            # Don't allow reverse-traversal of containment relationships
            if relation_type.lower() == "contain":
                continue
            step = [(neighbor, start_node, relation_type)]
            new_paths = _find_paths_dfs(model_set, neighbor, end_node, path + step, visited + [start_node])
            paths.extend(new_paths)

    # Return set of all paths from start_node to end_node
    return paths


def _find_paths_bfs(model_set, start_node, end_node, shortest_paths=False, max_steps=None):
    """Find all valid paths between `start_node` and `end_node` in the provided model set graph.
    Function explores paths through the graph in a breadth-first search (BFS) manner.
    :param model_set: Graph structure containing all model objects
    :param start_node: Start node.
    :param end_node: Destination node.
    :param shortest_paths: Option to return only those paths of shortest length from start_node to end_node.
    :return: List of models which can satisfy constraints of variable group and parent."""
    # Maintain list of explored edges to prevent traversal of the same edge in the same direction more than once
    path_queue = [{"next_node": start_node, "path": []}]
    valid_paths = []
    shortest_path_len = None

    while len(path_queue) > 0:
        # Pop the first item from the path queue for consideration
        q_item = path_queue.pop(0)
        node = q_item["next_node"]
        path = q_item["path"]

        # If the `shortest_paths` option is set, break iteration once we have considered all paths <= shortest_len
        # Since this is BFS, first path to exceed length indicates ALL remaining paths also exceed length.
        if shortest_paths and (shortest_path_len is not None) and (len(path) > shortest_path_len):
            break

        # If path exceeds the specified `max_steps`, stop search
        if (max_steps is not None) and (len(path) > max_steps):
            break

        # If current node and end_node are the same, this path is complete.
        # If this is the first completed path, set this as the shortest path length.
        if node == end_node:
            valid_paths.append(path)
            if shortest_path_len is None:
                shortest_path_len = len(path)
            continue

        # First explore all outgoing edges from start_node
        for edge_to_neighbor in model_set.out_edges(node, data="type"):
            # To prevent loops, don't allow an edge to be traversed multiple times in the same path
            if edge_to_neighbor not in path:
                _, neighbor, _ = edge_to_neighbor
                path_queue.append({"next_node": neighbor, "path": path + [edge_to_neighbor]})

        # Then explore all incoming edges from start_node (i.e. opposite of edge direction)
        for edge_to_neighbor in model_set.in_edges(node, data="type"):
            if edge_to_neighbor not in path:
                neighbor, _, relation_type = edge_to_neighbor
                # FIXME: This restriction is temporarily relaxed since WebGME models are missing some containment information
                #        e.g. BlueROV2 System model does not currently contain the hazard models or BTDs, or have an explicit relationship with them
                # # Don't allow reverse-traversal of containment relationships
                # if relation_type.lower() == "contain":
                #     continue
                path_queue.append({"next_node": neighbor, "path": path + [edge_to_neighbor]})

    return valid_paths


def search_models(model_set, meta_type, parent_node, shortest_paths=False):
    """Search model set to find all nodes of specified meta_type which are related to the parent_node.
    Return a list of matches where each match is a 2-tuple containing the matching node and the relationship path.
    If multiple paths exist between a matching node and the parent, return one match for each possible path."""

    # Find all nodes of the correct meta type
    candidate_nodes = []
    meta_type_lowercase  = [x.lower() for x in meta_type]
    for node_name, node_attrs in model_set.nodes(data=True):
        if node_attrs["type"].lower() in meta_type_lowercase:#meta_type.lower():
            candidate_nodes.append(node_name )

    # If no parent, then any node of the correct type is a valid match
    if parent_node is None:
        return [{"Name": node, "Path": None} for node in candidate_nodes]

    # Otherwise, find paths between each candidate node and the parent node.
    # Add one match for each node, path pair.
    matches = []
    for node in candidate_nodes:
        parent_paths = find_paths(model_set, parent_node, node, shortest_paths=shortest_paths, max_steps=10)
        if len(parent_paths) > 0:
            for path in parent_paths:
                matches.append({"Name": node, "Path": path})

    # If only the `shortest_paths` are desired, prune matches
    if shortest_paths:
        # Determine shortest path length
        shortest_path_len = sys.maxsize
        for match in matches:
            if len(match["Path"]) < shortest_path_len:
                shortest_path_len = len(match["Path"])

        # Keep only those entries with a path length <= shortest path length
        pruned_matches = []
        for match in matches:
            if len(match["Path"]) <= shortest_path_len:
                pruned_matches.append(match)
        matches = pruned_matches
    
    #print ('in search_models {0}'.format(matches[0]))

    return matches


if __name__ == "__main__":
    from models import example_model_set

    # Simple test cases
    print("Shortest paths from 'BlueROV' to 'ObstacleEncounterBTD':")
    print(find_paths(example_model_set, "BlueROV", "ObstacleEncounterBTD", shortest_paths=True))

    print("\nAll paths from 'BlueROV' to 'ObstacleEncounterBTD':")
    print(find_paths(example_model_set, "BlueROV", "ObstacleEncounterBTD"))

    print("\nObjects of type 'BTD' which are most closely related to the 'BlueROV' object:")
    for _match in search_models(example_model_set, "BTD", "BlueROV", shortest_paths=True):
        print(_match)

    print("\nAll objects of type 'BTD' which are related to the 'BlueROV' object:")
    for _match in search_models(example_model_set, "BTD", "BlueROV"):
        print(_match)
