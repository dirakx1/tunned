# This uses lamini api key
# export POWERML__PRODUCTION__KEY= 

import logging
from utilities import *
from llama import BasicModelRunner
from lamini import Lamini
import gradio as gr

logger = logging.getLogger(__name__)
global_config = None

# Train with lamini
import lamini

print (os.getenv("POWERML__PRODUCTION__KEY"))
lamini.api_key = os.getenv("POWERML__PRODUCTION__KEY")

#llm = Lamini(model_name='meta-llama/Meta-Llama-3-8B-Instruct')

## Using a trainned model 
llm = Lamini(model_name='6a2eabc6bd1bef03afd75e683d64f90f5f14ff436f62a64817d225975dd9776a')

#print(llm.generate("How are you?", output_type={"Response":"str"}))


#dataset_id = llm.upload_file("agroapp_docs.jsonl", input_key="question", output_key="answer")

#llm.train(data_or_dataset_id=dataset_id)

#llm.train(data_or_dataset_id='8741819033ad8888e89979529d182dc4fc517f7b30d871ab94e1f164f298074b')


def generate_response(prompt):
    response = llm.generate(prompt, output_type={"Response":"str"})
    return response

iface = gr.Interface(fn=generate_response, inputs="text", outputs="text", title="AgroApp AI")
iface.launch(share=True)
