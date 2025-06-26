---
layout: page
title: Fixing ASR
description: 2023-12
# img: assets/img/3.jpg
importance: 8
category: Work
---

At Amerandish, one of our key products is a Persian **ASR** system used by both individuals and businesses. Some business users, such as those in the pharmaceutical industry, required domain-specific ASR, especially for complex Persian medicine names.


At the time, our ASR system was based on **DeepSpeech2**, which incorporates an acoustic model and a language model (LM). We fine-tuned the LM to improve its predictions within specific domains by scraping relevant texts and retraining the LM with domain-specific data.


Additionally, we retrained the original LM after identifying output weaknesses. By eliminating typos in the text dataset and optimizing parameters in kenlm, we managed to **improve the word error rate by 20%**.
