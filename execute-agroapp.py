# conda info --envs
# conda activate slackllmbot
# This uses lamini api key
# export POWERML__PRODUCTION__KEY=
# python3.10 execute-agroapp.py 

import logging
from utilities import *
from lamini import Lamini
import gradio as gr

logger = logging.getLogger(__name__)
global_config = None

# Train with lamini
import lamini

#print (os.getenv("POWERML__PRODUCTION__KEY"))
#lamini.api_key = os.getenv("POWERML__PRODUCTION__KEY")
lamini.api_key = "7417748e68c7d1342652b741180fc452ba72452c0ed76f31f291a294fd4fe423"

## Using a trainned model 
llm = Lamini(model_name='eeaa90bfec9522740c7adf09da34e59da41c1c49db66fe3f4ddb21b54985f250')


def generate_response(prompt):
    response = llm.generate(prompt, output_type={"Response":"str"})
    return response

iface = gr.Interface(fn=generate_response,  inputs=[gr.Textbox(label="Question", lines=3)], outputs=[gr.Textbox(label="Answer", lines=3)], title="AgroApp AI")
iface.launch(share=False)
