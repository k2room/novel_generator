# ## Huggingface ###
# from transformers import GPT2Tokenizer, GPT2Model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2Model.from_pretrained('gpt2')
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)


### GPT2 without fine-tunning ###
import transformers
import torch
tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')

def gen_text(prompt_text, tokenizer, model, n_seqs=1, max_length=374):
    # n_seqs is the number of sequences to generate
    # max_length is the maximum length of the sequence
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=max_length+len(encoded_prompt),
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.2, # To ensure that we dont get repeated phrases
        do_sample=True,
        num_return_sequences=n_seqs
    )
    
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_() # the _ indicates that the operation will be done in-place
    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()
        text = tokenizer.decode(generated_sequence)
        total_sequence = (prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True, )) :])
        generated_sequences.append(total_sequence)
    
    return generated_sequences

temp = "I feel so unsure."
print(gen_text(temp,tokenizer,model))
