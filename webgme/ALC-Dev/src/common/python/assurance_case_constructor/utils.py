def node_to_tuple(node, name=None):
    #return name, node["rootId"], node["nodePath"]
    return name, node["nodePath"]


def tuple_to_node(_tuple, root_id):
    #name, root_id, node_path = _tuple
    #return {"rootId": root_id, "nodePath": node_path}
    name, node_path = _tuple
    return {"rootId": root_id,"nodePath": node_path}

