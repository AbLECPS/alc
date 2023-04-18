from pathlib import Path
import ipywidgets as widgets


class ImageVBox(widgets.VBox):

    widget_layout = widgets.Layout(
        display="flex", flex="1 1 0%", align_items="center",
        border="2px solid black"
    )

    button_layout = widgets.Layout(
        display="flex", flex="1 1 0%", justify_content="center", align_items="center",
        width="100%", margin="0 0 0 0"
    )

    button_class = "image-button"

    def toggle(self, button):
        if self.button_toggle:
            with self.image_file_path.open("rb") as image_fp:
                image_data = image_fp.read()
            image = widgets.Image(value=image_data, width=100, height=100)
            self.children = [self.button, image]
        else:
            self.children = [self.button]

        self.button_toggle = not self.button_toggle

    def __init__(self, image_file_path):

        widgets.VBox.__init__(self, layout=ImageVBox.widget_layout)

        if not isinstance(image_file_path, Path):
            image_file_path = Path(image_file_path)

        self.image_file_path = image_file_path
        image_file_name = self.image_file_path.name

        self.button = widgets.Button(description=image_file_name, layout=ImageVBox.button_layout)
        self.button.add_class(ImageVBox.button_class)

        self.button_toggle = False

        self.toggle(self.button)

        self.button.on_click(self.toggle)
