# conda activate slackllmbot
# This uses lamini api key
# export POWERML__PRODUCTION__KEY=
# python3.10 training-devops.py 

import os
import logging
from utilities import *

from lamini import Lamini
import lamini

logger = logging.getLogger(__name__)
global_config = None

# Train with lamini
print (os.getenv("POWERML__PRODUCTION__KEY"))
lamini.api_key = os.getenv("POWERML__PRODUCTION__KEY")

# input_key and ouptut_key must be related to jsonl file
llm = Lamini(model_name='meta-llama/Meta-Llama-3.1-8B-Instruct')
#dataset_id = llm.upload_file("datasets/devops_docs.jsonl", input_key="question", output_key="answer")

llm.train(data_or_dataset_id="b8a25c38b82badbde1ed7f879d90f84d977e91299254e887512a7d3dd81df804")


