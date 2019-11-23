from numpy import array
from os.path import join
from pandas import read_json
from random import sample
from math import floor
from sys import argv

FOLDER_NAME= 'Dataset'
TEST_FILE  = 'Sarcasm_Headlines_Dataset_v2_test.json'
TRAIN_FILE = 'Sarcasm_Headlines_Dataset_v2_train.json'

def split_sets(filename, test_sz=0.1):
    '''
        Splits dataset between TRAINING and TEST sets.
    '''
    
    # Reading json as pandas frame
    data = read_json(filename, 'records', lines=True)
    
    # Amount of examples
    amount = data.shape[0]
    
    # Getting test set from random positions and saving to CSV file
    test_idx = array(sample(range(amount), floor(amount*test_sz)))
    test = data.iloc[test_idx]
    test.to_json(join(FOLDER_NAME, TEST_FILE), 'records', lines=True)
    
    # Removing test data and saving training set to CSV file
    data.drop(test_idx, inplace=True)
    data.to_json(join(FOLDER_NAME, TRAIN_FILE), 'records', lines=True)

if __name__ == "__main__":
    split_sets(argv[1], float(argv[2]))
