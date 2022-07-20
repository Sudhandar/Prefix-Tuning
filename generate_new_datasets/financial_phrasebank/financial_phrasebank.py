import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset, DatasetDict
import pandas as pd
import collections


df = pd.read_csv('Sentences_AllAgree.txt', sep = '/t', encoding = 'latin-1', header = None, names = ['sentence'])
df['label'] = df['sentence'].str.split('@')
df['sentence'] = df['label'].apply(lambda x:x[0])
df['label'] = df['label'].apply(lambda x:x[1])
df['sentence'] = df['sentence'].str.strip()
df['label'] = df['label'].str.strip()
print(df['label'].value_counts())
dataset_arrow = Dataset(pa.Table.from_pandas(df))
dataset_arrow = dataset_arrow.class_encode_column("label")
# 90% train, 20% test + validation
train_testvalid = dataset_arrow.train_test_split(test_size=0.2, seed = 8, stratify_by_column = 'label')
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


train_df = train_test_valid_dataset['train'].to_pandas()
train_df = train_df[['sentence','label']].drop_duplicates()
valid_df = train_test_valid_dataset['test'].to_pandas()
valid_df = valid_df[['sentence','label']].drop_duplicates()
test_df = train_test_valid_dataset['validation'].to_pandas()
test_df = test_df[['sentence','label']].drop_duplicates()

train_df.to_csv('all_agree_train.csv',index=False)
train_df.to_csv('all_agree_dev.csv',index=False)
train_df.to_csv('all_agree_test.csv',index=False)

train_test_valid_dataset.save_to_disk("financial_phrasebank_all_agree.hf")