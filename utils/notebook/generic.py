from collections import defaultdict

from IPython.core.display import display, HTML


def full_width_notebook():
    html_str = "<style>.container { width:99% !important; }\n"
    html_str += "div.cell.selected { border-left-width: 1px !important; }\n"
    html_str += "div.output_scroll { resize: vertical !important }</style>"
    display(HTML(html_str))


def code_cell_toggle_button():
    html_str =  "<script>code_show=true;\n"
    html_str += "function code_toggle() {\n"
    html_str += "  if (code_show){\n"
    html_str += "    $('div.input').hide();\n"
    html_str += "  } else {\n"
    html_str += "    $('div.input').show();\n"
    html_str += "  }\n"
    html_str += "  code_show = !code_show\n"
    html_str += "}\n"
    html_str += "$( document ).ready(code_toggle);\n"
    html_str += "</script>\n"
    html_str += "<form action='javascript:code_toggle()'>\n"
    html_str += "<input type='submit' value='Click here to toggle on/off the raw code.'>\n"
    html_str += "</form>"
    display(HTML(html_str))


def notebook_input_prompt(variable_name, default_text="", button_label="OK [MUST PRESS]"):
    html_str = """
    <input id="inptval" style="width:60%;" type="text" value="DEFAULT_TEXT">
    <button onclick="set_value()" style="width:20%;">BUTTON_LABEL</button>

    <script type="text/Javascript">
        function set_value(){
            var input_value = document.getElementById('inptval').value;
            var command = "VARIABLE_NAME = '" + input_value + "'";
            var kernel = IPython.notebook.kernel;
            kernel.execute(command);
        }
    </script>
    """

    html_str = html_str.replace("VARIABLE_NAME", variable_name)
    html_str = html_str.replace("DEFAULT_TEXT", default_text)
    html_str = html_str.replace("BUTTON_LABEL", button_label)

    display(HTML(html_str))


def df_col_styler(col_colors=None):
    # Pandas dataframe styler. Each columns will have a color defined by 'col_colors'
    default_style = "text-transform: capitalize;"

    def apply_style(x):
        # copy df to new - original data are not changed
        df = x.copy()

        for i in range(len(df.columns)):
            if col_colors:
                color = f"rgba({col_colors[i][0]},{col_colors[i][1]},{col_colors[i][2]}, 0.6)"
                style = f"{default_style} background-color: {color};"
            else:
                style = default_style
            df[i] = style

        return df

    return apply_style


def separate_preds_ground_truth(processed_predictions, attribute=None):

    predictions = defaultdict(list)
    ground_truths = defaultdict(list)

    for processed_prediction in processed_predictions:
        if attribute:
            value = processed_prediction[attribute]
            predictions[value].append(processed_prediction['prediction'])
            ground_truths[value].append(processed_prediction['ground_truth'])

        predictions['all'].append(processed_prediction['prediction'])
        ground_truths['all'].append(processed_prediction['ground_truth'])

    if attribute is None:
        predictions = predictions['all']
        ground_truths = ground_truths['all']

    return predictions, ground_truths


def format_epoch_folder(epoch_folder):
    if type(epoch_folder) == int or epoch_folder.isdigit():
        epoch_folder = f"Epoch_{epoch_folder:02d}"

    assert epoch_folder.startswith('Epoch') or epoch_folder == 'best', "Invalid epoch folder provided"

    return epoch_folder
