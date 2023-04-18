from pathlib import Path
from functools import reduce


class RobustnessDataset:
    category_dict_key = "category_map"

    def __init__(self, data_dir_list, search_pattern_list=None, **kwargs):
        if search_pattern_list is None:
            search_pattern_list = ["**/*.png"]

        self.category_dict = kwargs.get(RobustnessDataset.category_dict_key, {})
        self.search_pattern_list = search_pattern_list

        if not isinstance(data_dir_list, list):
            data_dir_list = [data_dir_list]

        self.data_dir_list = data_dir_list

        self.image_file_list = None

    def get_category_number(self, image_file_relative_path, category_name):

        if category_name in self.category_dict:
            return self.category_dict.get(category_name)
        else:
            image_file_path = image_file_relative_path.absolute()
            if len(self.category_dict) == 0:
                message = "Could not get numeric category for image file \"{0}\":" \
                          "  pass in \"category_dict\" keyword argument to translate category name" \
                          " \"{1}\" to numeric category".format(image_file_path, category_name)
            else:
                message = "Could not get numeric category for image file \"{0}\":" \
                          "  category name \"{1}\" not in category_dict".format(
                    image_file_path, category_name
                )

            raise RuntimeError(message)

    @staticmethod
    def get_category_name(image_file_relative_path):
        return image_file_relative_path.parts[0]

    def fill_image_file_list(self):
        self.image_file_list = []
        for data_dir in self.data_dir_list:
            data_dir_path = Path(data_dir).absolute()
            for search_pattern in self.search_pattern_list:
                for image_file_path in data_dir_path.glob(search_pattern):
                    image_file_relative_path = image_file_path.relative_to(data_dir_path)
                    category_name = RobustnessDataset.get_category_name(image_file_relative_path)
                    category_number = self.get_category_number(
                        image_file_relative_path, category_name
                    )

                    self.image_file_list.append((image_file_path, category_name, category_number))

    def get_category_dict(self):

        category_set = set()
        for data_dir in self.data_dir_list:
            for category_dir in Path(data_dir).iterdir():
                if category_dir.is_dir():
                    category_set.add(category_dir.name)

        if reduce(lambda a, b: a and b, map(lambda x: x.isnumeric(), category_set)):
            category_list = sorted(map(lambda x: int(x), category_set))
            category_dict = {
                str(category_name): category_index + 1 for category_index, category_name in enumerate(category_list)
            }
        else:
            category_list = sorted(category_set)
            category_dict = {
                category_name: category_index + 1 for category_index, category_name in enumerate(category_list)
            }

        return category_dict

    def __len__(self):
        if self.image_file_list is None:
            self.fill_image_file_list()

        """Return the length of the loaded dataset. Function not required if using IterableDataset class."""
        len(self.image_file_list)

    def __getitem__(self, idx):
        if self.image_file_list is None:
            self.fill_image_file_list()

        """Fetch and return the training data point corresponding to index 'idx'.
        Data point should be returned as a 2-tuple in the form (input_data_point, output_data_point).
        Function not required if using IterableDataset class."""
        # Data must be properly formatted for training before returning from __getitem__.
        # See DataFormatter skeleton for usage details. General usage example:
        #
        # formatted_input_data = self.formatter.format_input(<raw_data_point>)
        # formatted_output_data = self.formatter.format_training_output(<raw_data_point>)
        # return formatted_input_data, formatted_output_data
        return self.image_file_list[idx]

    def __del__(self):
        """Perform any necessary cleanup when this class is deleted."""
        pass

    # Uncomment this function if using torch.utils.data.IterableDataset type dataset. Otherwise, can be removed.
    # def __iter__(self):
    #     """Return the next data point in the loaded dataset.
    #     Data point should be returned as a 2-tuple in the form (input_data_point, output_data_point).
    #     Function only required if using IterableDataset class"""
    #     pass
    # Define additional functions here as needed.

    def __iter__(self):
        if self.image_file_list is None:
            self.fill_image_file_list()

        for item in self.image_file_list:
            yield item
