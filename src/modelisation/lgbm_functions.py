
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import plotly.graph_objects as go


def Find_Nb_Estimators(data:pd.DataFrame,list_nb_estimators:list):

    data.dropna(inplace=True)
    target = data.columns[0]
    num_feat = [f for f in data.columns[1:] if data.dtypes[f]==np.float64]
    cat_feat = [f for f in data.columns[1:] if data.dtypes[f]==object]

    if len(cat_feat)>0:
        num_prepro = StandardScaler()
        cat_prepro = OneHotEncoder(handle_unknown='ignore')
        col_transformer =  ColumnTransformer([('num',num_prepro,num_feat),('cat',cat_prepro, cat_feat)])
        model = Pipeline([("coltrans",col_transformer),("lgbm",LGBMRegressor())])

    else:
        num_prepro = StandardScaler()
        model = Pipeline([("coltrans",num_prepro),("lgbm",LGBMRegressor())])

    
    list_mean_squared  = list()

    for nest in list_nb_estimators:

        hp_param_set = {
                "lgbm__n_estimators": nest
        }

        model.set_params(**hp_param_set)
        train_data_set = data.copy()
        train_x = train_data_set.drop(target,axis=1)
        train_y = train_data_set[target]
        model.fit(train_x,train_y)
        preds_train = model.predict(train_x)  
        rmse_train = mean_squared_error(train_y, preds_train,squared=False)
        list_mean_squared.append(rmse_train)

    df_meansquared = pd.DataFrame(data={'mean squared error':list_mean_squared,'nb estimateurs':list_nb_estimators})
    import plotly.graph_objects as go

    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=df_meansquared['nb estimateurs'], y=df_meansquared['mean squared error'],
                        mode='lines+markers'))


    fig.show()