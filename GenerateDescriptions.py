# allow users to pass in simple json with descriptions and call ollama llama3 to add detailed descriptions and produce an output json file - call via huggingface

import torch
import json
from transformers import AutoTokenizer, AutoModelForCasualLM

class GenerateDescriptions:
    def __init__(self, hf_token, model_id="meta-llama/Meta-Llama-3-8B"):
        pass