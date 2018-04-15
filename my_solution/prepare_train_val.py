from dataset import data_path
from sklearn.cross_validation import train_test_split
import pandas as pd

TRAIN_DISCRIPTION_FILE = 'train_df.csv'




def get_split(fold):
    
    train_df = pd.read_csv(str(data_path / TRAIN_DISCRIPTION_FILE) , sep=';')

    train, test = train_test_split(train_df.index, test_size=fold)

    val_file_names =list(train_df.loc[test, 'img_id'].values)

    train_file_names = list(train_df.loc[train, 'img_id'].values)

    return train_file_names, val_file_names
