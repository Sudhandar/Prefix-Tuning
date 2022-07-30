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



param_list_100 = {
    'aug_char_min': 20, 
    'aug_word_min': 20, 
    'aug_word_p': 1, 
    'aug_char_p': 1, 
    'aug_word_max': 50, 
    'aug_char_max': 50,
    'stopwords_regex': '[0-9]',
}

param_list_50 = {
    'aug_char_min': 10, 
    'aug_word_min': 10, 
    'aug_word_p': 1, 
    'aug_char_p': 1, 
    'aug_word_max': 20, 
    'aug_char_max': 20,
    'stopwords': ['a'],
    # 'stopwords': ['a','is','an','the','be','of','and','will','up','to'],
    'stopwords_regex': '[0-9]',
}

param_list_20 = {
    'aug_char_min': 2, 
    'aug_word_min': 2, 
    'aug_word_p': 0.2, 
    'aug_char_p': 0.2, 
    'aug_word_max': 10, 
    'aug_char_max': 10,
    'stopwords': ['a','is','an','the','be','of','and','will','up','to'],
    'stopwords_regex': '[0-9]',
}


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
        
    def random_character_insertion(self, percentage, word_percentage = 'default'):

        if word_percentage == 'default':
            aug = nac.RandomCharAug(action = 'insert',aug_char_min = 5, aug_word_min = 10,stopwords= ['a','is','an','the','be','of','and'], stopwords_regex= '[0-9]')        
        elif word_percentage == 0.2:
            aug = nac.RandomCharAug(action = 'insert', 
                aug_char_min = param_list_20['aug_char_min'],
                aug_word_min = param_list_20['aug_word_min'],
                aug_word_p = param_list_20['aug_word_p'],
                aug_char_p = param_list_20['aug_char_p'],
                aug_word_max = param_list_20['aug_word_max'],
                aug_char_max = param_list_20['aug_char_max'],
                stopwords = param_list_20['stopwords'],
                stopwords_regex = param_list_20['stopwords_regex'])

        elif word_percentage == 0.5:
            aug = nac.RandomCharAug(action = 'insert',
                aug_char_min = param_list_50['aug_char_min'],
                aug_word_min = param_list_50['aug_word_min'],
                aug_word_p = param_list_50['aug_word_p'],
                aug_char_p = param_list_50['aug_char_p'],
                aug_word_max = param_list_50['aug_word_max'],
                aug_char_max = param_list_50['aug_char_max'],
                stopwords = param_list_50['stopwords'],
                stopwords_regex = param_list_50['stopwords_regex'])

        elif word_percentage == 1:
            aug = nac.RandomCharAug(action = 'insert',
                aug_char_min = param_list_100['aug_char_min'],
                aug_word_min = param_list_100['aug_word_min'],
                aug_word_p = param_list_100['aug_word_p'],
                aug_char_p = param_list_100['aug_char_p'],
                aug_word_max = param_list_100['aug_word_max'],
                aug_char_max = param_list_100['aug_char_max'],
                stopwords_regex = param_list_100['stopwords_regex'])

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

    def qwerty_replacement(self, percentage, word_percentage):

        if word_percentage == 'default':
            aug = nac.KeyboardAug( aug_char_min = 5, aug_word_min = 5, aug_word_p = 0.5, aug_char_p = 0.5 )
        elif word_percentage == 0.2:
            aug = nac.KeyboardAug(aug_char_min = param_list_20['aug_char_min'],
                aug_word_min = param_list_20['aug_word_min'],
                aug_word_p = param_list_20['aug_word_p'],
                aug_char_p = param_list_20['aug_char_p'],
                aug_word_max = param_list_20['aug_word_max'],
                aug_char_max = param_list_20['aug_char_max'],
                stopwords = param_list_20['stopwords'],
                stopwords_regex = param_list_20['stopwords_regex'])

        elif word_percentage == 0.5:
            aug = nac.KeyboardAug( aug_char_min = param_list_50['aug_char_min'],
                aug_word_min = param_list_50['aug_word_min'],
                aug_word_p = param_list_50['aug_word_p'],
                aug_char_p = param_list_50['aug_char_p'],
                aug_word_max = param_list_50['aug_word_max'],
                aug_char_max = param_list_50['aug_char_max'],
                stopwords = param_list_50['stopwords'],
                stopwords_regex = param_list_50['stopwords_regex'])

        elif word_percentage == 1:
            aug = nac.KeyboardAug(aug_char_min = param_list_100['aug_char_min'],
                aug_word_min = param_list_100['aug_word_min'],
                aug_word_p = param_list_100['aug_word_p'],
                aug_char_p = param_list_100['aug_char_p'],
                aug_word_max = param_list_100['aug_word_max'],
                aug_char_max = param_list_100['aug_char_max'],
                stopwords_regex = param_list_100['stopwords_regex'])

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

    def ocr_replacement(self, percentage, word_percentage = 'default'):
        
        if word_percentage == 'default':
            aug = nac.OcrAug(aug_char_min = 5, aug_word_min = 5, aug_word_p=0.8)
        elif word_percentage == 0.2:
            aug = nac.OcrAug(aug_char_min = param_list_20['aug_char_min'],
                aug_word_min = param_list_20['aug_word_min'],
                aug_word_p = param_list_20['aug_word_p'],
                aug_char_p = param_list_20['aug_char_p'],
                aug_word_max = param_list_20['aug_word_max'],
                aug_char_max = param_list_20['aug_char_max'],
                stopwords = param_list_20['stopwords'],
                stopwords_regex = param_list_20['stopwords_regex'])

        elif word_percentage == 0.5:
            aug = nac.OcrAug(aug_char_min = param_list_50['aug_char_min'],
                aug_word_min = param_list_50['aug_word_min'],
                aug_word_p = param_list_50['aug_word_p'],
                aug_char_p = param_list_50['aug_char_p'],
                aug_word_max = param_list_50['aug_word_max'],
                aug_char_max = param_list_50['aug_char_max'],
                stopwords = param_list_50['stopwords'],
                stopwords_regex = param_list_50['stopwords_regex'])

        elif word_percentage == 1:
            aug = nac.OcrAug(aug_char_min = param_list_100['aug_char_min'],
                aug_word_min = param_list_100['aug_word_min'],
                aug_word_p = param_list_100['aug_word_p'],
                aug_char_p = param_list_100['aug_char_p'],
                aug_word_max = param_list_100['aug_word_max'],
                aug_char_max = param_list_100['aug_char_max'],
                stopwords_regex = param_list_100['stopwords_regex'])
            
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
corrupt_train_df = corrupter.random_character_insertion(1, word_percentage = 1)
corrupt_train_dataset = Dataset(pa.Table.from_pandas(corrupt_train_df))
corrupt_train_dataset = corrupt_train_dataset.class_encode_column("label")


valid_df = train_test_valid_dataset['test'].to_pandas()
valid_df = valid_df[['sentence','label']].drop_duplicates()
corrupter = Corruption(valid_df)
corrupt_valid_df = corrupter.random_character_insertion(1, word_percentage = 1)

corrupt_valid_dataset = Dataset(pa.Table.from_pandas(corrupt_valid_df))
corrupt_valid_dataset = corrupt_valid_dataset.class_encode_column("label")

train_test_valid_corrupt = DatasetDict({
    'train': corrupt_train_dataset,
    'test': test_valid['test'],
    'validation': corrupt_valid_dataset})

train_test_valid_corrupt.save_to_disk("./corrupt_data/random_character_insertion/financial_phrasebank_corrupt_100.hf")


test_df = train_test_valid_dataset['validation'].to_pandas()
test_df = test_df[['sentence','label']].drop_duplicates()

corrupt_train_df.to_csv('./corrupt_data/random_character_insertion/combined_corrupt_train_100.csv',index=False)
corrupt_valid_df.to_csv('./corrupt_data/random_character_insertion/combined_corrupt_dev_100.csv',index=False)

# train_df.to_csv('combined_train.csv',index=False)
# valid_df.to_csv('combined_dev.csv',index=False)
test_df.to_csv('./corrupt_data/random_character_insertion/combined_test.csv',index=False)


