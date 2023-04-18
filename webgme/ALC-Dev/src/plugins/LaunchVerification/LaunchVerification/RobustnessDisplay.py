import IPython.display as ipydis


def set_style():
    style_string = """
<style>
table.results {
    border:3px solid black;
    font-family:"Courier New", Courier, monospace;
    font-size: 14px;
}
th.image-path {
    border:2px solid black;
    text-align:center;
}
th.robustness-value {
    border:2px solid black;
    white-space:nowrap;
    text-align:center;
}
td.image-path {
    text-align:left;
    white-space:nowrap;
    border:1px solid black;
}
td.robustness-value {
    text-align:center;
    white-space:nowrap;
    border:1px solid black;
}
</style>"""

    return ipydis.HTML(style_string)


def display(robustness_results):
    html_string = '<table class="results">{}</table>'.format(
        '<th class="image-path">Image Path</th><th class="robustness-value">Robustness Value</th>' +
        ''.join(
            ['<tr>{}</tr>'.format(
                '<td class="image-path">{0}</td><td class="robustness-value">{1}</td>'.format(*row)
            ) for row in robustness_results]
        ))
    ipydis.display(ipydis.HTML(html_string))
