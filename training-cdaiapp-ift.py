# conda info --envs
# conda activate nameofvenen
# This uses lamini api key
# export API_KEY=
# python3.10 training-cdaiapp-ift.py 

import logging
import os
from utilities import *
from lamini import Lamini
import gradio as gr

logger = logging.getLogger(__name__)
global_config = None

# Train with lamini
from lamini import Lamini 
import jsonlines 
import lamini

print (os.getenv("API_KEY"))
lamini.api_key = os.getenv("API_KEY")

## Using a foundation model 
llm = Lamini(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct") 

def make_question(obj): 
     question = ( 
         f"Based on the devops docs, whats the answer to the question: {obj['question']}. " 
     ) 
     return question

def load_training_data(): 
     path = "datasets/devops_docs.jsonl" 
  
     limit = 10 
  
     with jsonlines.open(path) as reader: 
         for index, obj in enumerate(reader): 
             if index >= limit: 
                 break 
  
             header = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>" 
             header_end = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>" 
  
             yield { 
                 "input": header + make_question(obj) + header_end, 
                 "output": obj["answer"] + "<|eot_id|>", 
             }


dataset = list(load_training_data()) * 10 
  
llm.train( 
    data_or_dataset_id=dataset, 
        finetune_args={ 
             "max_steps": 300, 
             "early_stopping": False, 
             "load_best_model_at_end": False, 
        }, 
     ) 

