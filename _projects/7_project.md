---
layout: page
title: Language Model Training from Scratch
description: 2023-05
# img: assets/img/3.jpg
importance: 7
category: Work
---

Most of our tasks at work revolved around **text classification**, but BERT wasn’t always sufficient, especially when dealing with over 2,000 labels. Additionally, the data used to train BERT was becoming outdated, and language evolves constantly. We needed a new model to better classify text and address our needs.


We chose DeBERTa, one of the top models in benchmarks like SuperGLUE. DeBERTa uses an efficient training method inspired by ELECTRA’s Replaced Token Detection (RTD), involving a generator to corrupt tokens and a discriminator to identify them.


What this project taught me was the crucial role of data in training language models. I knew data was important, but I gained a deeper understanding of how diverse data from various domains, preprocessing, and tokenizer training can significantly impact model performance. I also dove deeper into transformer architectures and their underlying mechanisms.
