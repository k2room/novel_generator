import torch
import numpy as np
import json
import argparse
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

output_dir = './model_save/'

device = torch.device("cuda")
model = GPT2LMHeadModel.from_pretrained(output_dir)
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
model.to(device)
# generate samples
model.eval()

##################################################################
# Change the prompt(first sentence)
# prompt = "A man is sitting next to the river, and he trying to find something."
prompt = "a little girl standing in front of a picnic table"
##################################################################


generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
generated = generated.to(device)
print(generated)
sample_outputs = model.generate(
                                generated, 
                                #bos_token_id=random.randint(1,30000),
                                do_sample=True,   
                                top_k=50, 
                                max_length = 800,
                                top_p=0.95, 
                                num_return_sequences=4
                                )
print(" ")
for i, sample_output in enumerate(sample_outputs): 
    print("{}: {}\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))