import ipywidgets as widgets
from ResultHBox import ResultHBox


class ResultsVBox(widgets.VBox):

    image_path_key = "image_path"
    category_name_key = "category_name"
    category_number_key = "category_number"
    result_key = "result"

    widget_layout = widgets.Layout(display="flex", flex="1 1 0%", margin="0 0 0 0")

    def __init__(self, results_list):

        widgets.VBox.__init__(self)

        self.layout = ResultsVBox.widget_layout

        item_list = [ResultHBox.get_header()] + [
            ResultHBox(
                result.get(ResultsVBox.image_path_key),
                result.get(ResultsVBox.category_name_key),
                result.get(ResultsVBox.category_number_key),
                result.get(ResultsVBox.result_key)
            ) for result in results_list
        ]

        self.children = item_list
