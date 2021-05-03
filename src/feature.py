from datetime import date
import pandas as pd
from utils import load_config

config = load_config()

train = pd.read_csv(config['Train_Config']['path'])
test = pd.read_csv(config['Test_Config']['path'])

def extract_feature(df):
    df["published_timestamp"] = pd.to_datetime(df.published_timestamp).dt.date
    df["days_since_published"] = (date.today() - df.published_timestamp).dt.days
    return df

data_train = extract_feature(train)
data_test = extract_feature(test)

features = config['Train_Config']['features']
train_features = train[features]
test_features = test[features]


train_features.to_csv(config['Train_Config']['feature_path'], index = False)
test_features.to_csv(config['Test_Config']['feature_path'], index= False)