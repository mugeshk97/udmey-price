import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_config


config = load_config()


data = pd.read_csv(config['Config']['data_source'])

train, test = train_test_split(data, test_size = 0.2, random_state = config['Config']['random_seed'])


train.to_csv(config['Train_Config']['path'], index=False)
test.to_csv(config['Test_Config']['path'], index=False)