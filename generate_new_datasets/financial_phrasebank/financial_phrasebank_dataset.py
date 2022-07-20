import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset, DatasetDict
import pandas as pd
import collections
import glob

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
valid_df = train_test_valid_dataset['test'].to_pandas()
valid_df = valid_df[['sentence','label']].drop_duplicates()
test_df = train_test_valid_dataset['validation'].to_pandas()
test_df = test_df[['sentence','label']].drop_duplicates()

train_df.to_csv('combined_train.csv',index=False)
train_df.to_csv('combined_dev.csv',index=False)
train_df.to_csv('combined_test.csv',index=False)

train_test_valid_dataset.save_to_disk("financial_phrasebank.hf")