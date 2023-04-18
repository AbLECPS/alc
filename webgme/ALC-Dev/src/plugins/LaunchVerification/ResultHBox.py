import ipywidgets as widgets
from ImageVBox import ImageVBox


class ResultHBox(widgets.HBox):

    result_map = {
        0: "Failure",
        1: "Success",
        2: "Unknown"
    }

    image_path_heading_class = "image-path-heading"
    image_path_heading = "Image Path"
    image_path_heading_layout = widgets.Layout(
        display="flex", flex="1 1 0%", width="auto", height="100%",
        justify_content="center", align_items="center", border="3px solid black",
        margin="0 0 0 0"
    )

    category_name_heading_class = "category-name-heading"
    category_name_heading = "Category Name"
    category_name_heading_layout = widgets.Layout(
        display="flex", flex="1 1 0%", width="auto", height="100%",
        justify_content="center", align_items="center", border="3px solid black",
        margin="0 0 0 0"
    )

    category_name_class = "category-name"
    category_name_layout = widgets.Layout(
        display="flex", flex="1 1 0%", width="auto", height="100%",
        justify_content="center", align_items="center", border="2px solid black",
        margin="0 0 0 0"
    )

    category_number_heading_class = "category-number-heading"
    category_number_heading = "Category Number"
    category_number_heading_layout = widgets.Layout(
        display="flex", flex="1 1 0%", width="auto", height="100%",
        justify_content="center", align_items="center", border="3px solid black",
        margin="0 0 0 0"
    )

    category_number_class = "category-number"
    category_number_layout = widgets.Layout(
        display="flex", flex="1 1 0%", width="auto", height="100%",
        justify_content="center", align_items="center", border="2px solid black",
        margin="0 0 0 0"
    )

    result_heading_class = "result-heading"
    result_heading = "Result"
    result_heading_layout = widgets.Layout(
        display="flex", flex="1 1 0%", width="auto", height="100%",
        justify_content="center", align_items="center", border="3px solid black",
        margin="0 0 0 0"
    )

    result_class = "result"
    result_layout = widgets.Layout(
        display="flex", flex="1 1 0%", width="auto", height="100%",
        justify_content="center", align_items="center", border="2px solid black",
        margin="0 0 0 0"
    )

    widget_layout = widgets.Layout(
        display="flex", flex="1 1 0%", width="auto", margin="0 0 0 0"
    )

    @staticmethod
    def get_header():
        widget_1 = widgets.Label(ResultHBox.image_path_heading, layout=ResultHBox.image_path_heading_layout)
        widget_1.add_class(ResultHBox.image_path_heading_class)

        widget_2 = widgets.Label(ResultHBox.category_name_heading, layout=ResultHBox.category_name_heading_layout)
        widget_2.add_class(ResultHBox.category_name_heading_class)

        widget_3 = widgets.Label(ResultHBox.category_number_heading, layout=ResultHBox.category_number_heading_layout)
        widget_3.add_class(ResultHBox.category_number_heading_class)

        widget_4 = widgets.Label(ResultHBox.result_heading, layout=ResultHBox.result_heading_layout)
        widget_4.add_class(ResultHBox.result_heading_class)

        retval = widgets.HBox()
        retval.children = [widget_1, widget_2, widget_3, widget_4]

        return retval

    def __init__(self, image_file_path, category_name, category_number, result):

        widgets.HBox.__init__(self)

        self.layout = ResultHBox.widget_layout

        widget_1 = ImageVBox(image_file_path)

        widget_2 = widgets.Label(str(category_name), layout=ResultHBox.category_name_layout)
        widget_2.add_class(ResultHBox.category_name_class)

        widget_3 = widgets.Label(str(category_number), layout=ResultHBox.category_number_layout)
        widget_3.add_class(ResultHBox.category_number_class)

        widget_4 = widgets.Label(ResultHBox.result_map.get(int(result)), layout=ResultHBox.result_layout)
        widget_4.add_class(ResultHBox.result_class)

        self.children = [ widget_1, widget_2, widget_3, widget_4 ]
