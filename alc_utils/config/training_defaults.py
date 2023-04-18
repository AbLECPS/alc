import alc_utils.data_types
from . import config_util

# What percent of data should be used for training & testing. Rest is discarded.
USEFUL_DATA_FRACTION = 1.00
# What percent of data should be used for training. Rest is used for validation.
TRAINING_DATA_FRACTION = 1.00
EPOCHS = 5
BATCH_SIZE = 64
RNG_SEED = 10
SHUFFLE = False
DATASET_NAME = "rosbag"
USE_GENERATORS = False
TRAIN_ASSURANCE_MONITOR = False
ASSURANCE_MONITOR_TYPE = 'svdd'
DATA_SPLIT_MODE = alc_utils.data_types.SplitMode.INTER_BATCH
DATA_BATCH_MODE = alc_utils.data_types.BatchMode.BATCH_SIZE
VERBOSE = True
UPLOAD_RESULTS = False
LOSS = "mse"
OPTIMIZER = "adam"
METRICS = ["accuracy"]
CALLBACKS = []

# How often to calculate validation loss in number of epochs (1 == every epoch)
VALIDATE_INTERVAL = 1
# How often to print training statistics in number of iterations (1 == every iteration)
PRINT_INFO_INTERVAL = 10

var_dict = config_util.build_var_dict(globals())
var_dict_lower = config_util.build_var_dict(globals(), convert_case="lower")
