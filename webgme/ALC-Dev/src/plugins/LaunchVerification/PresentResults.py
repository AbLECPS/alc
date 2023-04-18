from pathlib import Path
from ResultsVBox import ResultsVBox
from IPython.display import display, HTML
import json


class PresentResults:

    def __init__(self, results_file_path):

        if not isinstance(results_file_path, Path):
            results_file_path = Path(results_file_path)

        self.results_file_path = results_file_path

    @staticmethod
    def get_style():
        return display(
            HTML(
"""<style>
    table.robustness-test {
        border:3px solid black;
        font-family:"Courier New", Courier, monospace;
        font-size: 14px;
    }
    th.robustness-title {
        text-align: center;
        border:2px solid black;
    }
    td.robustness-name {
        text-align: left;
        border:1px solid black;
    }    
    td.robustness-entry {
        text-align: left;
        border:1px solid black;
    }    
    table.parameter-table {
        border:3px solid black;
        font-family:"Courier New", Courier, monospace;
        font-size: 12px;
    }
    th.parameter-name-title {
        text-align: center;
        border:2px solid black;
    }
    td.parameter-name {
        text-align: left;
        border:2px solid black;
    }
    th.parameter-value-title {
        text-align: center;
        border:2px solid black;
    }
    td.parameter-value {
        text-align: right;
        border:2px solid black;
    }
    .image-path-heading {
        font-size: 18px;
        background-color: #AAEEEE;
        font-weight: bold;
    }
    .image-button {
        font-size: 18px
    }
    .category-name-heading {
        font-size: 18px;
        background-color: #AAEEEE;
        font-weight: bold;
    }
    .category-name {
        font-size: 18px
    }
    .category-number-heading {
        font-size: 18px;
        background-color: #AAEEEE;
        font-weight: bold;
    }
    .category-number {
        font-size: 18px
    }
    .result-heading {
        font-size: 18px;
        background-color: #AAEEEE;
        font-weight: bold;
    }
    .result {
        font-size: 18px
    }
</style>"""
            )
        )

    @staticmethod
    def sort_results_list(x):
        image_path = Path(x.get(ResultsVBox.image_path_key))
        return x.get(ResultsVBox.category_number_key), str(image_path.name)

    def get_results_list(self):
        with self.results_file_path.open("r") as results_fp:
            results_list = json.load(results_fp)

        results_list.sort(key=PresentResults.sort_results_list)
        return results_list

    @staticmethod
    def present_results(results_list):

        results_vbox = ResultsVBox(results_list)

        return display(results_vbox)
