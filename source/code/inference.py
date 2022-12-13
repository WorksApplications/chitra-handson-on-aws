import torch
import numpy as np
from transformers import BertForSequenceClassification
from sudachitra import BertSudachipyTokenizer

# ref: https://github.com/huggingface/notebooks/blob/main/sagemaker/17_custom_inference_script/sagemaker-notebook.ipynb

def model_fn(model_dir):
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertSudachipyTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def predict_fn(data, model_and_tok):
    model, tokenizer = model_and_tok

    sentence = data.pop("inputs", data)
    processed = tokenizer(sentence, padding=True, truncation=True,
                          max_length=512, return_tensors="pt")
    
    with torch.no_grad():
        model_output = model(**processed)

    logits = model_output.logits.tolist()
    output = {
        "label": np.argmax(logits),
        "logits": logits,
    }
    return output
