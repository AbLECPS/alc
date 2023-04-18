from traitlets import Set, Unicode
from nbconvert.preprocessors import ExecutePreprocessor


class TagRunPreprocessor(ExecutePreprocessor):
    execute_cell_tags = Set(Unicode(), default_value=[],
                            help=("Tags indicating which cells are to be executed,"
                                  "matches tags in `cell.metadata.tags`.")).tag(config=True)

    def preprocess(self, nb, resources=None, km=None):
        """
        Preprocessing to apply to each notebook. See base.py for details.
        """
        # Skip preprocessing if the list of patterns is empty
        if not bool(self.execute_cell_tags):
            self.log.info("No tags specified, so no cells executed")
            return nb, resources

        self.log.info("Executing only cells with tags", self.execute_cell_tags)
        return ExecutePreprocessor.preprocess(self, nb, resources, km)

    def preprocess_cell(self, cell, resources, cell_index, store_history=True):

        if not bool(self.execute_cell_tags.intersection(cell.get('metadata', {}).get('tags', []))):
            return cell, resources

        return ExecutePreprocessor.preprocess_cell(self, cell, resources, cell_index, store_history=store_history)
