# Assurance Monitor confidence message

# Enumeration of the supported types of assurance monitor
uint8 NULL_TYPE=0
uint8 TYPE_CONF_INTERVAL=1
uint8 TYPE_KNN=2

# Output info
uint8 type
float32[] confidence_levels
std_msgs/Float32MultiArray confidence_values