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
        samples_to_convert = int(can_be_converted.shape[0] * percentage)
        print('Expected conversions:', samples_to_convert)
        converted = can_be_converted.sample(n = samples_to_convert, random_state = 8)
        print('No. of examples converted:', converted.shape[0] )
        self.df = self.df[~self.df['sentence'].isin(converted['sentence'])]
        converted.pop('sentence')
        converted = converted[['new_sentence','label']]
        converted.columns = ['sentence','label']
        self.df = self.df.append(converted)
        print('Final Dataframe shape:', self.df.shape)
        return self.df
        
    def random_character_insertion(self, percentage):
        aug = nac.RandomCharAug(action = 'insert',aug_char_min = 5, aug_word_min = 10,stopwords= ['a','is','an','the','be','of','and'], stopwords_regex= '[0-9]')        
        # self.df['new_sentence'] = self.df['sentence'].apply(lambda x:aug.augment(x)[0])
        print('Dataframe shape:',self.df.shape)
        samples_to_convert = int(self.df.shape[0] * percentage)
        print('Expected conversions:', samples_to_convert)
        converted = self.df.sample(n = samples_to_convert, random_state = 8)
        converted['new_sentence'] = converted['sentence'].apply(lambda x:aug.augment(x)[0])
        print('No. of examples converted:', converted.shape[0] )
        self.df = self.df[~self.df['sentence'].isin(converted['sentence'])]
        converted.pop('sentence')
        converted = converted[['new_sentence','label']]
        converted.columns = ['sentence','label']
        self.df = self.df.append(converted)
        print('Final Dataframe shape:', self.df.shape)
        return self.df

    def random_word_deletion(self, percentage):
        aug = naw.RandomWordAug(action='delete', aug_p = 0.6 , aug_min = 5, stopwords= ['a','is','an','the','be','of','and'], stopwords_regex= '[0-9]')
        print('Dataframe shape:',self.df.shape)
        samples_to_convert = int(self.df.shape[0] * percentage)
        print('Expected conversions:', samples_to_convert)
        converted = self.df.sample(n = samples_to_convert, random_state = 8)
        converted['new_sentence'] = converted['sentence'].apply(lambda x:aug.augment(x)[0].strip())
        print('No. of examples converted:', converted.shape[0] )
        self.df = self.df[~self.df['sentence'].isin(converted['sentence'])]
        converted.pop('sentence')
        converted = converted[['new_sentence','label']]
        converted.columns = ['sentence','label']
        self.df = self.df.append(converted)
        print('Final Dataframe shape:', self.df.shape)
        self.df = self.df.sample(frac = 1, random_state = 8)
        return self.df

    def qwerty_replacement(self, percentage):

        # aug_char_min - 1, aug_word_p = 0.1, aug_word_min-1, - 3/21
        # The c*m9nQy also said %Mat it would lower the price of d#celopm#Mt projects by about one third compared with last November.

        # aug_char_min - 2, aug_word_p = 0.2, aug_word_min-2, - 6/21
        # The compnay also said tBa% it wi^ld lower the price of development 0roiecFs by about one third compared wiHb last gIFember.

        # aug_char_min - 5, aug_word_p = 0.5, aug_word_min-5, - 11/21
        # The V)npMa^ A?XL WQLE 6NQ4 it siI<w lower the oGLdR of development projects by xn9 T$ one third compared Eoym <xQG J(geKNer.

        # aug_char_min - 30, aug_word_p = 1, aug_word_min-30, aug_char_p -1, - 16/21 - No stop words 
        # The f8N)bWu Xpe9 Qzuf RJXG it QPkpe o*ed5 the (T*xs of x4DWi)0JShf OTPusdhD by zfIJ5 one %Go5w sk,ox$@R @Kgb IQZ6 BIgWkgfG.

        # aug = nac.KeyboardAug( aug_char_min = 30, aug_word_min =30, aug_word_p = 1 , aug_char_p =1, stopwords= ['a','is','an','the','be','of','and'], stopwords_regex= '[0-9]')
        aug = nac.KeyboardAug( aug_char_min = 30, aug_word_min = 30, aug_word_p = 1, aug_char_p = 1 )
        print('Dataframe shape:',self.df.shape)
        samples_to_convert = int(self.df.shape[0] * percentage)
        print('Expected conversions:', samples_to_convert)
        converted = self.df.sample(n = samples_to_convert, random_state = 8)
        converted['new_sentence'] = converted['sentence'].apply(lambda x:aug.augment(x)[0].strip())
        print('No. of examples converted:', converted.shape[0] )
        self.df = self.df[~self.df['sentence'].isin(converted['sentence'])]
        converted.pop('sentence')
        converted = converted[['new_sentence','label']]
        converted.columns = ['sentence','label']
        self.df = self.df.append(converted)
        print('Final Dataframe shape:', self.df.shape)
        self.df = self.df.sample(frac = 1, random_state = 8)
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

# train_test_valid_dataset.save_to_disk("financial_phrasebank.hf")

train_df = train_test_valid_dataset['train'].to_pandas()
train_df = train_df[['sentence','label']].drop_duplicates()
corrupter = Corruption(train_df)
corrupt_train_df = corrupter.qwerty_replacement(1)
corrupt_train_dataset = Dataset(pa.Table.from_pandas(corrupt_train_df))
corrupt_train_dataset = corrupt_train_dataset.class_encode_column("label")


valid_df = train_test_valid_dataset['test'].to_pandas()
valid_df = valid_df[['sentence','label']].drop_duplicates()
corrupter = Corruption(valid_df)
corrupt_valid_df = corrupter.qwerty_replacement(1)

corrupt_valid_dataset = Dataset(pa.Table.from_pandas(corrupt_valid_df))
corrupt_valid_dataset = corrupt_valid_dataset.class_encode_column("label")

train_test_valid_corrupt = DatasetDict({
    'train': corrupt_train_dataset,
    'test': test_valid['test'],
    'validation': corrupt_valid_dataset})

train_test_valid_corrupt.save_to_disk("./corrupt_data/qwerty_replacement_all/financial_phrasebank_corrupt_80.hf")


test_df = train_test_valid_dataset['validation'].to_pandas()
test_df = test_df[['sentence','label']].drop_duplicates()


corrupt_train_df.to_csv('./corrupt_data/qwerty_replacement_all/combined_corrupt_train_80.csv',index=False)
corrupt_valid_df.to_csv('./corrupt_data/qwerty_replacement_all/combined_corrupt_dev_80.csv',index=False)

# train_df.to_csv('combined_train.csv',index=False)
# valid_df.to_csv('combined_dev.csv',index=False)
test_df.to_csv('./corrupt_data/qwerty_replacement_all//combined_test.csv',index=False)