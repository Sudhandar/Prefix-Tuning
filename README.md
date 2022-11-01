# Exploring Robustness of Prefix Tuning in Noisy Data: A Case Study in Financial Sentiment Analysis

This github repo contains the code for the paper titled "Exploring Robustness of Prefix Tuning in Noisy Data: A Case Study in
Financial Sentiment Analysis". This paper has been accepted at the Financial NLP (FinNLP) workshop at the EMNLP conference 2022. The full version of the paper will be published after December 9, 2022.

### Overview

The financial phrasebank dataset is corrupted using various text corruption methods such as keyboard errors (typos), inserting random characters, deleting random words, replacing characters with OCR alternatives and replacing words with antonyms by varying percentages in each sentence. The corrupted dataset is used with two widely used pre-trained models, BERT-base and RoBERTa-large, under both prefix tuning and fine-tuning, to compare their performance at different noise levels. In addition, the performance is also evaluated on the Kaggle Stock Market Tweets dataset, which is a real-life noisy dataset. The results for the experiments will be made available to the public once the paper has been published.

## Methodology

![alt text](https://github.com/Sudhandar/Prefix-Tuning/blob/main/images/Prefixtuning_bert.png)

## Corruption module

The corruption module consists of 5 text corruption methods which closely replicate the noise found in real-world data. This module is used to corrupt the clean financial dataset and the corrupted dataset is used to evaluate the performance of the models. The following are the various text corruption methods used in the corruption module. The nlpaug library is used for generating the various corruption methods.


1. **Keyboard Error (QWERTY)** Simulates typing mistakes made while using a QWERTY-type keyboard.
2. **Random Character Insertion** Inserts random characters into a word in a sentence.
3. **Random Word Deletion** Randomly deletes a word from the sentence.
4. **OCR Replacement** Replaces the characters in the word with their OCR equivalents, e.g., stock can be replaced as st0ck (here an alphabet, o, is replaced with the number zero, 0)
5. **Antonym Replacement** Replaces the words with their antonyms (opposite meaning) in the sentence.

## Dataset

### Financial phrasebank dataset

The Financial Phrasebank dataset, consists of 4840 sentences from financial news articles and the sentences were manually labelled as positive, negative or neutral by 16 annotators with backgrounds in finance and business. The annotators labelled the sentences depending on whether the information from the sentence had a positive, negative or no impact on the stock prices of the company mentioned in the sentence. It is an imbalanced dataset with 1363 positive sentences, 604 negative sentences and 2873 neutral sentences. In addition to it, depending on the agreement level among the annotators on the polarity of the sentence, the dataset was classified into 50%, 66%, 75% and 100% agreement levels. For example, 50% annotator agreement means more than 50% of the annotators agreed and selected the same polarity for a particular sentence. The financial phrasebank dataset with 50% annotator agreement level (4840 sentences) is used to run the experiments on estimating the robustness of prefix tuning and the 100% agreement level (2262 sentences) is used to compare the performance of the prefix tuning and fine tuning without adding any noise.

## Implementation details

The experiments were carried out on four Nvidia GeForce RTX 2080 GPU's for 30 epochs. After experiments, a prefix length of 20 was used to evaluate the performance of the models. The learning rate differs for each model and method. For prefix tuning, both BERT-base and RoBERTa-large models use a learning rate of 1e-2. For fine-tuning, BERT-base uses a learning rate of 2e-5 and RoBERTa-large used a learning rate of 2e-6.

