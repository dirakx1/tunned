import os
from llama import BasicModelRunner
import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import time
import torch
import transformers
import pandas as pd
import jsonlines

from utilities import *

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
from llama import BasicModelRunner
from lamini import Lamini

logger = logging.getLogger(__name__)
global_config = None

# Train with lamini
import lamini
lamini.api_key = ""

# input_key and ouptut_key must be related to jsonl file
llm = Lamini(model_name='meta-llama/Meta-Llama-3-8B-Instruct')
dataset_id = llm.upload_file("devops_docs_1.jsonl", input_key="question", output_key="answer")

llm.train(data_or_dataset_id=dataset_id)


