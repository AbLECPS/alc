import networkx as nx

example_model_set = nx.DiGraph()
# models.add_node("TEMP", type="System")
# models.add_edge("TEMP", "BlueROV", type="contain")
example_model_set.add_node("BlueROV", type="System")

# Hazards
example_model_set.add_node("HazardSet", type="HazardDescription")
example_model_set.add_node("ObstacleEncounter", type="Hazard")
example_model_set.add_node("DeviationFromArea", type="Hazard")
example_model_set.add_node("LossOfPipeline", type="Hazard")

# BTDs
example_model_set.add_node("ObstacleEncounterBTD", type="BTD")
example_model_set.add_node("DeviationFromAreaBTD", type="BTD")
example_model_set.add_node("LossOfPipelineBTD", type="BTD")

# Functions
example_model_set.add_node("CommandAuthority", type="Function")
example_model_set.add_node("ObstacleDetection", type="Function")
example_model_set.add_node("AvoidanceLogic", type="Function")

# Relationships
example_model_set.add_edge("BlueROV", "HazardSet", type="contain")
example_model_set.add_edge("HazardSet", "ObstacleEncounter", type="contain")
example_model_set.add_edge("HazardSet", "DeviationFromArea", type="contain")
example_model_set.add_edge("HazardSet", "LossOfPipeline", type="contain")
example_model_set.add_edge("BlueROV", "ObstacleEncounterBTD", type="contain")
example_model_set.add_edge("BlueROV", "CommandAuthority", type="contain")
example_model_set.add_edge("BlueROV", "ObstacleDetection", type="contain")
example_model_set.add_edge("BlueROV", "AvoidanceLogic", type="contain")
example_model_set.add_edge("ObstacleEncounterBTD", "ObstacleEncounter", type="describe")
example_model_set.add_edge("ObstacleEncounterBTD", "CommandAuthority", type="require")
example_model_set.add_edge("ObstacleEncounterBTD", "AvoidanceLogic", type="require")
example_model_set.add_edge("ObstacleEncounterBTD", "ObstacleDetection", type="require")
