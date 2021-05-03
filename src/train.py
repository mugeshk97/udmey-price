import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from utils import load_config, write_json, save_model


config = load_config()

data = pd.read_csv(config['Train_Config']['feature_path'])

x = data.drop(config['Train_Config']['target'], axis =1)
y = data[config['Train_Config']['target']]

model = RandomForestRegressor(
    n_estimators=config['RandomForest']['estimators'], max_depth=config['RandomForest']['depth'],random_state=config['Config']['random_seed']
)

model = model.fit(x, y)

paramerters = {'n_estimators': config['RandomForest']['estimators'], 'max_depth': config['RandomForest']['depth']}

write_json(config['Report']['params_path'], paramerters)

save_model(config['Model']['path'], model)
