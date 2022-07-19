import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset, DatasetDict
import pandas as pd
import collections


df = pd.read_csv('stock_data.csv')
df = df[['Text','Sentiment']].drop_duplicates()
df.columns = ['sentence','label']
df.loc[df['label'] < 0, 'new_label'] = 'negative'
df.loc[df['label'] > 0, 'new_label'] = 'positive'
df = df[['sentence','new_label']].drop_duplicates()
df.columns = ['sentence','label']
df['sentence'] = df['sentence'].str.strip()
df['label'] = df['label'].str.strip()
df = df.drop_duplicates()
print(df['label'].value_counts())

dataset_arrow = Dataset(pa.Table.from_pandas(df))
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
print('Train dataset distribution:')
print(collections.Counter(train_test_valid_dataset['train']['label']))
print('Validation dataset distribution:')
print(collections.Counter(train_test_valid_dataset['test']['label']))
print('Test dataset distribution:')
print(collections.Counter(train_test_valid_dataset['validation']['label']))
train_test_valid_dataset.save_to_disk("kaggle_tweets.hf")