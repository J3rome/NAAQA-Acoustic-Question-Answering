from collections import defaultdict
import random

from IPython.core.display import display, HTML


def full_width_notebook():
    html_str = "<style>.container { width:99% !important; margin-left:auto !important; margin-right:auto !important; }\n"
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


def notebook_input_prompt(variable_name, default_text="", button_label="CHOOSE", default_answer=None, selected=False):
    html_str = """
    <input id="INPUT_ID" style="INPUT_STYLE" type="text" value="DEFAULT_TEXT">
    <button onclick="INPUT_ID()" style="width:20%;">BUTTON_LABEL</button>

    <script type="text/Javascript">
        document.getElementById('INPUT_ID').addEventListener("keyup", function(event) {
            if (event.keyCode == 13){
                event.preventDefault();
                event.target.nextElementSibling.click()
            }
        })
    
        function INPUT_ID(){
            var input_box = document.getElementById('INPUT_ID');
            var default_text = "DEFAULT_TEXT";
            if(input_box.value == default_text){
                var command = "VARIABLE_NAME = ('" + input_box.value + "', DEFAULT_ANSWER)"
            }else{
                var command = "VARIABLE_NAME = ('" + input_box.value + "', None)"
            }
            
            var kernel = IPython.notebook.kernel;
            kernel.execute(command);
            
            // Clear the colors & Assign color to chosen
            var all_inputs = document.querySelectorAll('*[id^="inptval"]');
            
            all_inputs.forEach(function(node){
                node.style.backgroundColor = '';
            })
            
            input_box.style.backgroundColor = 'orange';
        }
    </script>
    """

    input_style = "width:60%;"
    if selected:
        input_style += "background-color:orange;"

    html_str = html_str.replace("INPUT_STYLE", input_style)
    html_str = html_str.replace("VARIABLE_NAME", variable_name)
    html_str = html_str.replace("DEFAULT_TEXT", default_text)
    if default_answer is None:
        default_answer = 'None'
    else:
        default_answer = f"'{default_answer}'"
    html_str = html_str.replace("DEFAULT_ANSWER", default_answer)
    html_str = html_str.replace("BUTTON_LABEL", button_label)
    html_str = html_str.replace("INPUT_ID", f"inptval_{random.randint(0,10000)}")

    display(HTML(html_str))


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
