class UniqueNumberGenerator:

    def __init__(self):
        self.unique_number = 0

    def get_unique_number(self):
        retval = self.unique_number
        self.unique_number += 1
        return retval
