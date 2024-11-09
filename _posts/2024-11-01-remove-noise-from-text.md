---
layout: post
title: Removing Noise from ASR Text (Complete Code Walkthrough)
date: 2024-11-01 15:09:00
description: This guide shows how to filter meaningless noise like 'um' and 'aaa' from text, improving quality for NLP tasks.
tags: noise_filtering text_cleaning
categories: NLP
featured: true
---

<div class="row mt-3">
    <div class="col-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/blog/2024/cleaning_in_progress.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Image by Oliver Hale @ unsplash.com
</div>

We’ve all been there. You’re trying to make sense of a conversation recorded during a phone call or transcribed from a voice assistant, and what you get is littered with noise: “uhh”, “em”, or endless repetitions of “aaa.” It’s frustrating to sift through the gibberish, and it makes automated text processing difficult. This challenge often arises when dealing with Automatic Speech Recognition (ASR) systems.

Think about it—ASR systems are not perfect. They try their best to convert speech into text, but background noise, filler words, or even accents can introduce meaningless "noise" into the transcription. If you’re building an AI application to handle customer queries, or analyze voice calls, you need a way to filter out that junk, or your models might get confused by non-informative words. 

Now, imagine if we could automatically clean up such noisy transcriptions, removing meaningless noise while keeping the relevant text intact. Let’s walk through a solution for this problem, where we can filter out noisy tokens from a text by leveraging pre-trained transformers and masking.

### The Solution: Leveraging Masked Language Models

We’ll use a masked language model to help us identify which parts of the text are meaningful and which can be considered noise. The idea is to mask words one by one and have the model predict what should be in that position. If the model is highly confident that the masked word is correct (based on the context), we’ll keep it; otherwise, it’s likely to be noise and can be discarded.

Here’s how we do it, step by step:

1. **Tokenizing the Text**: We first break down the text into smaller units, called tokens. These tokens could be words, parts of words, or even single characters.
  
2. **Masking One Token at a Time**: For each token, we mask it—essentially hiding it from the model—and ask the model to predict what should be there. The model will give us a probability distribution of possible tokens for that masked position.

3. **Filtering Based on Confidence**: If the model’s confidence (or probability) for the original token is high, we assume the token is useful. If the confidence is low, it’s probably noise and we discard it.

4. **Reconstructing the Text**: Once we’ve filtered out the noisy tokens, we convert the valid tokens back into a clean string.

Let’s look at the code that implements this.

<br />


### The Code

```python
from transformers import PreTrainedTokenizerFast, AutoModelForMaskedLM
import torch
model_name='google-bert/bert-base-uncased' # you can use any MLM model from Huggingface

tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = torch.tensor([token_ids])

# Mask each token one by one and get the model's prediction for the masked token
def filter_tokens_with_threshold(input_ids, tokens, threshold=0.001):
    valid_tokens = []
    
    for i, token_id in enumerate(input_ids[0]):
        
        masked_input_ids = input_ids.clone()
        masked_input_ids[0, i] = tokenizer.mask_token_id  # Mask the current token
        
        # Get model predictions for the masked token
        with torch.no_grad():
            outputs = model(masked_input_ids)
        
        logits = outputs.logits[0, i]
        
        probabilities = torch.softmax(logits, dim=-1)
        
        token_prob = probabilities[token_id].item()
        
        # Keep the token if its probability is above the threshold
        if token_prob > threshold:
            valid_tokens.append(tokens[i])
    
    return valid_tokens

filtered_tokens = filter_tokens_with_threshold(input_ids, tokens, threshold=0.0002)

filtered_text = tokenizer.convert_tokens_to_string(filtered_tokens)

print(f"Filtered text: {filtered_text}")
```
<br />
<br />

### Breaking Down the Code

**1. Loading the Pre-trained Model and Tokenizer**
We begin by loading a pre-trained tokenizer and a masked language model (MLM) from the Hugging Face library. The `model_name` would be a model that suits your language needs (for example, `bert-base-uncased` for English).

```python
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
```
<br />

**2. Tokenizing the Input Text**
Next, we break the input text into tokens. These tokens are then converted into token IDs, which is how the model understands the text.

```python
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
```
<br />

**3. Creating Input Tensors**
We convert the list of token IDs into a tensor, which is the format expected by the PyTorch-based model.

```python
input_ids = torch.tensor([token_ids])
```
<br />

**4. Masking Tokens and Predicting**
Here’s where the magic happens. For each token, we temporarily mask it and ask the model to predict what should be in its place. The model’s output gives us probabilities for every possible token. If the model's prediction for the original token is above a set threshold, we keep the token; otherwise, it’s discarded as noise.

```python
masked_input_ids[0, i] = tokenizer.mask_token_id  # Mask the current token

outputs = model(masked_input_ids)
logits = outputs.logits[0, i]
probabilities = torch.softmax(logits, dim=-1)
token_prob = probabilities[token_id].item()

if token_prob > threshold:
    valid_tokens.append(tokens[i])
```
<br />

**5. Filtering with a Threshold**
We define a threshold to decide which tokens to keep. This threshold can be fine-tuned depending on how strict we want the filtering to be. Lower thresholds may retain more tokens, while higher thresholds may strip away more noise.

**6. Reconstructing the Clean Text**
Finally, we take the remaining valid tokens and convert them back into a readable string format.

```python
filtered_text = tokenizer.convert_tokens_to_string(filtered_tokens)
```
<br />
<br />

### Why This Works

Masked language models, like BERT, are trained to predict missing words based on the context around them. When we mask a token in our noisy text and ask the model to predict the token that should be there, the model uses the surrounding context to make a highly informed guess. If it confidently predicts that a token should be in its place, we trust that the token is meaningful. If not, it’s probably just noise, and we can safely remove it.

<br />

### The Result: Cleaner Text

Using this method, we can filter out non-informative parts of a transcription, improving the quality of the text data. This is especially useful for downstream tasks like sentiment analysis, intent detection, or even summarization, where cleaner input can lead to more accurate outputs.

For example, if our noisy input is:
```
"Hi, uh, I aaa want to, um, order a pizza."
```
The filtered output might be:
```
"Hi, I want to order a pizza."
```

This approach can be extended and refined with larger datasets, adjusted thresholds, or custom fine-tuned models to adapt to specific domains like customer service or healthcare.

<br />

### Conclusion

As voice-based applications grow, dealing with noisy text is more important than ever. By using masked language models, we can filter out noise and improve the quality of transcriptions for better AI performance.

I'm Amir Hossein Karimi, an NLP Engineer passionate about helping machines understand humans. If you found this useful, check out my other blogs for more on AI and natural language processing!
