# conda info --envs
# conda activate nameofvenen
# This uses lamini api key
# export API__KEY=
# python3.10 execute-cdaiapp.py 

import logging
import os
from utilities import *
from lamini import Lamini
import gradio as gr

logger = logging.getLogger(__name__)
global_config = None

# Train with lamini
import lamini

#print (os.getenv("POWERML__PRODUCTION__KEY"))
lamini.api_key = "7417748e68c7d1342652b741180fc452ba72452c0ed76f31f291a294fd4fe423"


## Using a trainned model 
llm = Lamini(model_name='e5f3eda8e5275a33781f47c622d0a6f1aa552455ec8203b89f468afa83bfade0')


def generate_response(prompt):
    response = llm.generate(prompt, output_type={"Response":"str"})
    return response

iface = gr.Interface(fn=generate_response, allow_flagging=False, inputs=[gr.Textbox(label="Question", lines=3)], outputs=[gr.Textbox(label="Answer", lines=3)], title="Devops AI")
iface.launch(share=False)
