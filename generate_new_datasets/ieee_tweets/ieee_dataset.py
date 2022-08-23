import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset, DatasetDict
import pandas as pd
import collections
import re

mentions = re.compile(r"^@\S+|\s@\S+")
urls = re.compile(r"https?://\S+")

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, ' ', data)

def process_text(text):
    text = re.sub(r'http\S+', ' ', text)
    text = mentions.sub(' ', text)
    return text.strip().lower()

df = pd.read_csv('ieee_tweets_labelled.csv', delimiter = ';')
df = df.dropna()
df = df[['text','sentiment']].drop_duplicates()
df.columns = ['sentence','label']
df['sentence'] = df.sentence.apply(process_text)
df['sentence'] = df.sentence.apply(remove_emojis)
df['sentence'] = df['sentence'].str.replace('\n',' ')
df['sentence'] = df['sentence'].str.replace('rt',' ')
df['sentence'] = df['sentence'].str.strip()
df['label'] = df['label'].str.strip()
df = df.drop_duplicates()
print(df['label'].value_counts())

dataset_arrow = Dataset(pa.Table.from_pandas(df))
dataset_arrow = dataset_arrow.class_encode_column("label")
# 70% train, 30% test + validation
train_testvalid = dataset_arrow.train_test_split(test_size=0.3, seed = 8, stratify_by_column = 'label')
# Split the 30% test + valid in half test, half valid
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

train_df.to_csv('train.csv',index=False)
valid_df.to_csv('dev.csv',index=False)
test_df.to_csv('test.csv',index=False)

train_test_valid_dataset.save_to_disk("ieee_tweets.hf")