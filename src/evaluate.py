from sklearn.metrics import mean_squared_error
import pandas as pd
from utils import load_config, write_json, load_model

config = load_config()

test = pd.read_csv(config['Test_Config']['feature_path'])

x = test.drop(config['Train_Config']['target'], axis =1)
y = test[config['Train_Config']['target']]

model = load_model(config['Model']['path'])

y_pred = model.predict(x)


mse = mean_squared_error(y, y_pred)

report = {'mse': mse}

write_json(config['Report']['metrics_path'] , report)
