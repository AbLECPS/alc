# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
"""Definition of various data-types used by the ALC utility classes."""


# NOTE: Previously used python "Enum" class for enumerated data types, but this leads to issue with JSON serialization
class BatchMode:
    """Defines if/how loaded data should be divided into batches.
        0) NONE == Do not subdivide data into sets. Return one large set of all data.
        1) FILE == Divide data into sets corresponding to each file loaded.
        2) BATCH_SIZE == Divide data into sets equal to the specified batch size."""
    NONE = 0
    FILE = 1
    BATCH_SIZE = 2


class SplitMode:
    """Defines if/how loaded data should be split into testing/training sets.
        1) INTRA_BATCH == Randomly split datapoints within each batch into sets.
        2) INTER_BATCH == Randomly select each batch as training/testing. Do not split datapoints within a batch."""
    INTRA_BATCH = 1
    INTER_BATCH = 2
