# conda info --envs
# conda activate slackllmbot
# This uses lamini api key
# export POWERML__PRODUCTION__KEY=
# python3.10 training-agroapp.py 

import logging
from utilities import *
from lamini import Lamini
import gradio as gr

logger = logging.getLogger(__name__)
global_config = None

# Train with lamini
import lamini

print (os.getenv("POWERML__PRODUCTION__KEY"))
lamini.api_key = os.getenv("POWERML__PRODUCTION__KEY")


## Using a trainned model 
llm = Lamini(model_name='6a2eabc6bd1bef03afd75e683d64f90f5f14ff436f62a64817d225975dd9776a')


def generate_response(prompt):
    response = llm.generate(prompt, output_type={"Response":"str"})
    return response

iface = gr.Interface(fn=generate_response, inputs="text", outputs="text", title="AgroApp AI")
iface.launch(share=True)
