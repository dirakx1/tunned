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


# basic model runner
llm = Lamini(model_name='meta-llama/Meta-Llama-3.1-8B-Instruct')

#dataset_id = llm.upload_file("datasets/agroapp_docs_total.jsonl", input_key="question", output_key="answer")

#llm.train(data_or_dataset_id=dataset_id)

llm.train(data_or_dataset_id='8741819033ad8888e89979529d182dc4fc517f7b30d871ab94e1f164f298074b')
