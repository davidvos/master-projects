# Master Thesis in NLG at DPG Media

## Introduction

Lots of news follows a predictable pattern. Examples are the reports on local sport matches. This project would investigate (semi-)automating the production of such news. You would have our whole archive of millions of historical news articles available to train on. - DPG Media

The field of research is called 'Data-to-Text'. The challenge is to take structured information (like a table or a graph) and use it to generate a sequence of words. The  

## Deadlines

 * [Complete Thesis Contract](https://datanose.nl/#yourprojects/instance[84004]/approval) (07/01/2022)
 * Complete Thesis Proposal (07/03/2022)
 * Submit Thesis (05/07/2022)

More information on practicalities can be found [here](https://student.uva.nl/ai/content/az/master-thesis-ai/master-thesis-ai-2020.html).

## Related Work

These are the most important papers related to my thesis. 

[Text-to-Text Pre-Training for Data-to-Text Tasks](https://aclanthology.org/2020.inlg-1.14.pdf)
Introduces fine-tuning for Data-to-Text tasks using T5.

[Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353.pdf)
Introduces prefix-tuning as a more efficient way of fine-tuning for data-to-text tasks.

[Control Prefixes for Text Generation](https://arxiv.org/abs/2110.08329)
Extends prefix-tuning so it is able to condition on input-based metadata.

[[Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing]](https://arxiv.org/pdf/2107.13586.pdf?ref=https://githubhelp.com)
Survey on the field of prompt-learnin / prefix-tuning methods.

## Main Todos

#### Multilingual Prefixes
 * Transfer method to T5 language model
 * Finetune full T5 model on English/Russian WebNLG
 * Train prefix with multilingual model on English/Russian WebNLG 
 
 #### More structural approach towards prefix tuning
 * Read into GCN/translation methods
 * Integrate GCN in prefix tuning method
 * Train baseline prefix tuning method on WebNLG
 * Train GCN prefix tuning method on WebNLG

 #### Creation Dutch soccer dataset
 * Complete and extend web scraper for statistics
 * Create script that connects statistics to summaries
 * Format dataset to usable (WebNLG-like) format