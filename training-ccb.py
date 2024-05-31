import logging
from utilities import *
from llama import BasicModelRunner


logger = logging.getLogger(__name__)
global_config = None

# Train with lamini
model = BasicModelRunner("meta-llama/Meta-Llama-3-8B-Instruct") 
model.load_data_from_jsonlines("devops_docs_1.jsonl", input_key="question", output_key="answer")
model.train(is_public=True) 


