import pandas as pd
import glob
from sklearn.model_selection import train_test_split


class FinancialPhrasebankPandas():
  def __init__(self, type):
    self.type = type
    self.main_df = pd.DataFrame()

    if self.type == 'all':
      all_files = glob.glob("*.txt")
      combined_df = pd.DataFrame()

      print('Files considered:', all_files)
      for file in all_files:
        df = pd.read_csv(file, sep = '/t', encoding = 'latin-1', header = None, names = ['sentence'])
        df = self.get_labels(df)
        combined_df = combined_df.append(df)
        combined_df = combined_df.drop_duplicates()
      self.main_df = combined_df.copy()
      print(self.main_df.shape)

    elif self.type == 'all_agree':
        df = pd.read_csv('Sentences_AllAgree.txt', sep = '/t', encoding = 'latin-1', header = None, names = ['sentence'])
        self.main_df = self.get_labels(df)


    train, valid_test = train_test_split(self.main_df, test_size=0.3, stratify = self.main_df['label'], random_state = 8)
    print(train.shape)
    valid, test = train_test_split(valid_test, test_size=0.5, stratify = valid_test['label'], random_state= 8)
    print(valid.shape)
    print(test.shape)

    train_name  = self.type + 'train.csv'
    valid_name  = self.type + 'valid.csv'
    test_name  = self.type + 'test.csv'

    train.to_csv(train_name,  index = False)
    valid.to_csv(valid_name,  index = False)
    test.to_csv(test_name, index = False)


  def convert_labels(self, df):
    '''
    df - encodes label (string - negative, neutral and positive) into integer (0,1,2)
    returns - df with two columns sentence and label
    '''
    '''
    df - Contains a single column named 'sentence' which contains both the labels and the sentence
    returns - df with two columns sentence and label
    '''
    df.loc[df['label'] =='negative','new_label'] = 0
    df.loc[df['label'] =='neutral','new_label'] = 1
    df.loc[df['label'] =='positive','new_label'] = 2
    df = df[['sentence','new_label']]
    df.columns = ['sentence','label']
    df['label'] = df['label'].apply(int)

    return df

  def get_labels(self, df):
    '''
    df - Contains a single column named 'sentence' which contains both the labels and the sentence
    returns - df with two columns sentence and label
    '''
    df['label'] = df['sentence'].str.split('@')
    df['sentence'] = df['label'].apply(lambda x:x[0])
    df['label'] = df['label'].apply(lambda x:x[1])
    df['sentence'] = df['sentence'].str.strip()
    df['label'] = df['label'].str.strip()
    df = self.convert_labels(df)
    df = df.drop_duplicates()

    return df

if __name__ == '__main__':
  FinancialPhrasebankPandas('all')
  FinancialPhrasebankPandas('all_agree')

