# Assurance Monitor confidence message

# Enumeration of the supported types of assurance monitor
uint8 NULL_TYPE=0
uint8 TYPE_CONF_INTERVAL=1
uint8 TYPE_KNN=2
uint8 TYPE_SVDD=3

# Output info
uint8 type
float32 confidence_level
float32[] values