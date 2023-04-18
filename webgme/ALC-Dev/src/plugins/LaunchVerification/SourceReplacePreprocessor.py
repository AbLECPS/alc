import re
from traitlets import CRegExp, Tuple, List, Unicode, Set, Bool
from nbconvert.preprocessors import Preprocessor


class SourceReplacePreprocessor(Preprocessor):

    source_replace_tags = Set(Unicode(), default_value=[],
                              help=("Tags indicating which cells are to have their source replaced,"
                                    "matches tags in `cell.metadata.tags`.")).tag(config=True)

    regex_substitution_list = List(Tuple(CRegExp(), Unicode(), Bool()))

    def preprocess(self, nb, resources=None, km=None):
        """
        Preprocessing to apply to each notebook. See base.py for details.
        """
        # Skip preprocessing if the list of patterns is empty
        if not bool(self.source_replace_tags):
            self.log.info("No tags specified, so no cells will have their source replaced")
            return nb, resources

        self.log.info("Replacing source only in cells with tags", self.source_replace_tags)
        return Preprocessor.preprocess(self, nb, resources)

    def preprocess_cell(self, cell, resources, index):

        if not bool(self.source_replace_tags.intersection(cell.get('metadata', {}).get('tags', []))):
            return cell, resources

        source = cell.source
        for regex, replacement, replace_all_flag in self.regex_substitution_list:
            if bool(re.search(regex, source)):
                cell.source = replacement if replace_all_flag else re.sub(regex, replacement, source)
                return cell, resources

        return cell, resources
