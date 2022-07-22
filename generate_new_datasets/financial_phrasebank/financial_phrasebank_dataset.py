#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 14:08:08 2022

@author: sudhandar
"""

import pyarrow as pa
# import pyarrow.dataset as ds
from datasets import Dataset, DatasetDict
import pandas as pd
import collections
import glob
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw

class Corruption:
    def __init__(self, df):
        self.df = df
    
    def antonym_generator(self, percentage):
        aug = naw.AntonymAug( aug_p = 0.5 , aug_min = 10, stopwords= ['a','is','an','the','be','of','and'], stopwords_regex= '[0-9]')
        self.df['new_sentence'] = self.df['sentence'].apply(lambda x:aug.augment(x)[0])
        no_antonyms = self.df[self.df['sentence']==self.df['new_sentence']]
        can_be_converted = self.df[self.df['sentence']!=self.df['new_sentence']]
        self.df.pop('new_sentence')
        print('Dataframe shape:',self.df.shape)
        print('Can be converted:', can_be_converted.shape)
        print('Cannot be converted:', no_antonyms.shape)
        samples_to_convert = int(can_be_converted.shape[0] * 0.2)
        print('Expected conversions:', samples_to_convert)
        converted = can_be_converted.sample(n = samples_to_convert, random_state = 8)
        print('No. of examples converted:', converted.shape[0] )
        self.df = self.df[~self.df['sentence'].isin(converted['sentence'])]
        converted.pop('sentence')
        converted.columns = ['sentence','label']
        self.df = self.df.append(converted)
        print('Final Dataframe shape:', self.df.shape)
        return self.df
        


all_files = glob.glob("*.txt")
print(all_files)
combined_df = pd.DataFrame()
for file in all_files:
  df = pd.read_csv(file, sep = '/t', encoding = 'latin-1', header = None, names = ['sentence'])
  df['label'] = df['sentence'].str.split('@')
  df['sentence'] = df['label'].apply(lambda x:x[0])
  df['label'] = df['label'].apply(lambda x:x[1])
  df['sentence'] = df['sentence'].str.strip()
  df['label'] = df['label'].str.strip()
  combined_df = combined_df.append(df)
  combined_df = combined_df.drop_duplicates()
  # print(combined_df['label'].value_counts())
  

dataset_arrow = Dataset(pa.Table.from_pandas(combined_df))
dataset_arrow = dataset_arrow.class_encode_column("label")
# 90% train, 20% test + validation
train_testvalid = dataset_arrow.train_test_split(test_size=0.3, seed = 8, stratify_by_column = 'label')
# Split the 20% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed = 8, stratify_by_column = 'label')
# gather everyone if you want to have a single DatasetDict
train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'validation': test_valid['train']})
print(combined_df.shape[0])
print('Train dataset distribution:')
print(collections.Counter(train_test_valid_dataset['train']['label']))
print('Validation dataset distribution:')
print(collections.Counter(train_test_valid_dataset['test']['label']))
print('Test dataset distribution:')
print(collections.Counter(train_test_valid_dataset['validation']['label']))



train_df = train_test_valid_dataset['train'].to_pandas()
train_df = train_df[['sentence','label']].drop_duplicates()
corrupter = Corruption(train_df)
corrupt_train_df = corrupter.antonym_generator(0.2)


valid_df = train_test_valid_dataset['test'].to_pandas()
valid_df = valid_df[['sentence','label']].drop_duplicates()
corrupter = Corruption(valid_df)
corrupt_valid_df = corrupter.antonym_generator(0.2)


test_df = train_test_valid_dataset['validation'].to_pandas()
test_df = test_df[['sentence','label']].drop_duplicates()


corrupt_train_df.to_csv('combined_corrupt_train.csv',index=False)
corrupt_valid_df.to_csv('combined_corrupt_dev.csv',index=False)
test_df.to_csv('combined_test.csv',index=False)


train_df.to_csv('combined_train.csv',index=False)
valid_df.to_csv('combined_dev.csv',index=False)
test_df.to_csv('combined_test.csv',index=False)

train_test_valid_dataset.save_to_disk("financial_phrasebank.hf")