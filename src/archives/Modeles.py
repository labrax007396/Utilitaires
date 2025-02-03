
import pandas as pd
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np


#def Run_Model(type:str, data:pd.DataFrame, split_test_train:bool, params:dict):



def Build_And_Train_Model(data:pd.DataFrame, model_type:str, split_test_train:bool, params:dict, dico_model:dict):


    if model_type == "LinearRegression":
        preproc        = Build_Preproc(data,dico_model)
        model          = Pipeline([("preprocessor",preproc),("model", LinearRegression())])
        model_result   = Run_Model(model,data,split_test_train,model_type,params,dico_model)


    if model_type == "LGBMRegressor":
        preproc        = Build_Preproc(data,dico_model)
        model          = Pipeline([("preprocessor",preproc),("model", LGBMRegressor())])
        params['metric'] = 'rmse'
        params['random_state'] = 48
        hp_param_set = {
            f"model__{key}": value for key, value in params.items()
        }
        model.set_params(**hp_param_set)
        model_result   = Run_Model(model,data,split_test_train,model_type,params,dico_model)

    return model_result
 

def Build_Preproc(data:pd.DataFrame, dico_model:dict):

    data.dropna(inplace=True)

    num_feat = [f for f in data.columns[1:] if data.dtypes[f]==np.float64]
    cat_feat = [f for f in data.columns[1:] if data.dtypes[f]==object]

    if len(cat_feat)>0:
        num_prepro = StandardScaler()

        ### On regarde si il n'y a que certaines catégories prises en compte
        dico_categories = dict()
        list_categories = list()

        for tag, d in dico_model['facteurs'].items():
            if d['type'] == 'cat':
                if 'categories' in d.keys():
                    dico_categories[d['nom']] = d['categories']

        for cf in cat_feat:
            if cf in dico_categories.keys():
                list_categories.append(dico_categories[cf])  

        if len(list_categories)>0:
            cat_prepro = OneHotEncoder(categories=list_categories,handle_unknown='ignore')
        else:
            cat_prepro = OneHotEncoder(handle_unknown='ignore')

        preproc =  ColumnTransformer([('num',num_prepro,num_feat),('cat',cat_prepro, cat_feat)])
    else:
        preproc = StandardScaler()

    return preproc



def Run_Model(model,data,split_test_train,model_type,params,dico_model):

    data.dropna(inplace=True)
    target = data.columns[0]   

    if split_test_train:
        train_data_set, test_data_set = train_test_split(data, test_size=0.33, random_state=42)
        train_x = train_data_set.drop(target,axis=1)
        train_y = train_data_set[target]
        model.fit(train_x,train_y)
        
    else:
        train_data_set = data.copy()
        test_data_set  = None
        train_x = train_data_set.drop(target,axis=1)
        train_y = train_data_set[target]
        model.fit(train_x,train_y)    

    R2_train, R2_test, mape_train, mape_test = Compute_Perf_Model(model,train_data_set,test_data_set)

    if model_type == "LinearRegression": 
        formula = Formula_RegLin(data=data,dico_model=dico_model)
    else:
        formula = ''

    metrics =   {'R2_train':R2_train,
                 'R2_test':R2_test,
                 'mape_train': mape_train, 
                 'mape_test': mape_test}

    fig_modele_mesure = Plot_ymod_ymes(model,train_data_set,test_data_set)


    model_result = {'model':model,
                    'model_type':model_type,
                    'metrics':metrics,
                    'params':params,
                    'formula':formula,
                    'train_data_set':train_data_set, 
                    'test_data_set':test_data_set,
                    'figure':fig_modele_mesure
                    }

    return model_result



def Compute_Perf_Model(model,train_data_set,test_data_set):

    target = train_data_set.columns[0]

    if test_data_set is not None:
        train_x = train_data_set.drop(target,axis=1)
        train_y = train_data_set[target]
        test_x  = test_data_set.drop(target,axis=1)
        test_y  = test_data_set[target]    
        y_train_modelise = model.predict(train_x)
        y_test_modelise  = model.predict(test_x)
        R2_train = r2_score(train_y, y_train_modelise)
        R2_test  = r2_score(test_y, y_test_modelise)
        mape_train =  100*mean_absolute_percentage_error(train_y, y_train_modelise)
        mape_test  =  100*mean_absolute_percentage_error(test_y, y_test_modelise)

    else:
        
        train_x = train_data_set.drop(target,axis=1)
        train_y = train_data_set[target]    
        y_train_modelise = model.predict(train_x)
        R2_train = r2_score(train_y, y_train_modelise)
        R2_test  = None
        mape_train =  100*mean_absolute_percentage_error(train_y, y_train_modelise)
        mape_test  =  None

    return R2_train, R2_test, mape_train, mape_test

def Plot_ymod_ymes(model,train_data_set,test_data_set):

    import matplotlib.pyplot as plt
    #%matplotlib inline
    # Turn interactive plotting off
    plt.ioff()

    fig = plt.figure(figsize=(8,8))
    

    target = train_data_set.columns[0]


    if test_data_set is not None:
        train_x = train_data_set.drop(target,axis=1)
        train_y = train_data_set[target]
        test_x  = test_data_set.drop(target,axis=1)
        test_y  = test_data_set[target]    
        y_train_modelise = model.predict(train_x)
        y_test_modelise  = model.predict(test_x)

        plt.scatter(train_y, y_train_modelise, marker= 'o', s=30, alpha=0.8,label='Apprentissage')
        plt.scatter(test_y, y_test_modelise, marker= 's',color='green', s=30, alpha=0.8,label='Validation')

    else:
        
        train_x = train_data_set.drop(target,axis=1)
        train_y = train_data_set[target]    
        y_train_modelise = model.predict(train_x)

        plt.scatter(train_y, y_train_modelise, marker= 'o', s=30, alpha=0.8,label='Apprentissage')

    plt.plot([0,train_y.max()], [0,train_y.max()], 'r-')


    
    plt.xlabel('Mesure')
    plt.ylabel('Modèle')
    plt.axis('equal')
    plt.axis([0,train_y.max(), 0, train_y.max()])
    plt.legend()
    #plt.show()

    return fig





def powerset(iterable):
        #"powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    from itertools import chain, combinations
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def Run_RegLin_All_Combination_Features(data:pd.DataFrame, p_value:float):

    import numpy as np
    import statsmodels.formula.api as smf

    df_models = pd.DataFrame(columns=['Features','R2'])
    iter_model = 0

    for subset in powerset(data.columns[1:].to_list()):
        if len(subset) > 0:
            #print(list(subset))
            features = ' + '.join(subset)
            formule_statmodel = data.columns[0]+ ' ~ '+features
            results = smf.ols(formule_statmodel, data=data).fit()

            if np.all(results.pvalues.values <= p_value):
                iter_model = iter_model + 1
                df_models.loc[iter_model,'Features'] = features
                df_models.loc[iter_model,'R2'] = results.rsquared

    df_models.sort_values(by=['R2'],inplace=True, ascending=False)

    return df_models

def Formula_RegLin(data:pd.DataFrame, dico_model:dict):

    data.dropna(inplace=True)
    target = data.columns[0]

    num_feat = [f for f in data.columns[1:] if data.dtypes[f]==np.float64]
    cat_feat = [f for f in data.columns[1:] if data.dtypes[f]==object]

    ### On regarde si il n'y a que certaines catégories prises en compte
    list_categories = list()

    for tag, d in dico_model['facteurs'].items():
        if d['type'] == 'cat':
            if 'categories' in d.keys():
                sub_list = [d['nom']+"_"+m for m in d['categories']]
                list_categories = list_categories + sub_list

    train_data_set = data.copy()
    train_x = train_data_set.drop(target,axis=1)
    train_y = train_data_set[target]

    if len(cat_feat)>0:

        X = pd.get_dummies(train_x,columns=cat_feat)

        ### On regarde si il n'y a que certaines catégories prises en compte
        list_categories = list()

        for tag, d in dico_model['facteurs'].items():
            if d['type'] == 'cat':
                if 'categories' in d.keys():
                    sub_list = [d['nom']+"_"+m for m in d['categories']]
                    list_categories = list_categories + sub_list

        if len(list_categories) > 0:
            X = X[num_feat+list_categories]

    else:
        X = train_x



    model_sansprepro = LinearRegression()
    model_sansprepro.fit(X ,train_y)

    formula = ''
    for f,c in zip(X.columns, model_sansprepro.coef_):
        if math.copysign(1,c) > 0.0:
            op = '+'
        else:
            op = ''

        formula = formula + ' ' + op + str(round(c,8)) + '*' + f

    if math.copysign(1,model_sansprepro.intercept_) > 0.0:
        op = '+'
    else:
        op = ''

    formula = formula + ' '  + op + str(round(model_sansprepro.intercept_,8))
    formula = formula[2:]


    if math.copysign(1,model_sansprepro.coef_[0]) < 0:
        formula = "-"+formula

    return formula


def Run_RegLin(data:pd.DataFrame, split_test_train:bool):

    data.dropna(inplace=True)
    target = data.columns[0]

    num_feat = [f for f in data.columns[1:] if data.dtypes[f]==np.float64]
    cat_feat = [f for f in data.columns[1:] if data.dtypes[f]==object]

    if len(cat_feat)>0:
        num_prepro = StandardScaler()
        cat_prepro = OneHotEncoder(handle_unknown='ignore')
        col_transformer =  ColumnTransformer([('num',num_prepro,num_feat),('cat',cat_prepro, cat_feat)])
        model = Pipeline([("preprocessor",col_transformer),("model", LinearRegression())])

    else:
        num_prepro = StandardScaler()
        model = Pipeline(steps=[("preprocessor", num_prepro),("model" , LinearRegression())])

    if split_test_train:
        train_data_set, test_data_set = train_test_split(data, test_size=0.33, random_state=42)
        train_x = train_data_set.drop(target,axis=1)
        train_y = train_data_set[target]
        #test_x  = test_data_set.drop(target,axis=1)
        #test_y  = test_data_set[target]
        model.fit(train_x,train_y)
        
        #y_train_modelise = model.predict(train_x)
        #y_test_modelise  = model.predict(test_x)
        #R2_train = r2_score(train_y, y_train_modelise)
        #R2_test  = r2_score(test_y, y_test_modelise)

    else:
        train_data_set = data.copy()
        test_data_set  = None
        train_x = train_data_set.drop(target,axis=1)
        train_y = train_data_set[target]
        model.fit(train_x,train_y)
        
        #y_train_modelise = model.predict(train_x)
        #y_test = y_test_modelise = None
        #R2_train = r2_score(train_y, y_train_modelise)
        #R2_test  = None

    R2_train, R2_test, mape_train, mape_test = Compute_Perf_Model(model,train_data_set,test_data_set)

    #if len(cat_feat)>0:
    #    cat_prepro = OneHotEncoder(handle_unknown='ignore')
    #    col_transformer =  ColumnTransformer([('cat',cat_prepro, cat_feat)])
            #model_sansprepro = Pipeline([("preprocessor",col_transformer),("model", LinearRegression())])
    #    X = col_transformer.fit_transform(train_x)
    #    X = np.asarray(X.todense())
    #    X = np.concatenate((X, train_x[num_feat].values), axis=1)
    #    f_name = model.steps[0][1].get_feature_names_out()
    #    f_name = [f.replace('num__','').replace('cat__','') for f in f_name]

    #    X = pd.DataFrame(index=train_x.index, data=X, columns=f_name)

    #else:
    #    X = train_x

    X = pd.get_dummies(train_x)
    model_sansprepro = LinearRegression()
    model_sansprepro.fit(X ,train_y)

    formula = ''
    for f,c in zip(X.columns, model_sansprepro.coef_):
        if math.copysign(1,c) > 0.0:
            op = '+'
        else:
            op = ''

        formula = formula + ' ' + op + str(round(c,8)) + '*' + f

    if math.copysign(1,model_sansprepro.intercept_) > 0.0:
        op = '+'
    else:
        op = ''

    formula = formula + ' '  + op + str(round(model_sansprepro.intercept_,8))
    formula = formula[2:]


    if math.copysign(1,model_sansprepro.coef_[0]) < 0:
        formula = "-"+formula

    metrics =   {'R2_train':R2_train,
                 'R2_test':R2_test,
                 'mape_train': mape_train, 
                 'mape_test': mape_test}

    fig_modele_mesure = Plot_ymod_ymes(model,train_data_set,test_data_set)


    model_result = {'model':model,
                    'model_type':'Regression lineaire',
                    'metrics':metrics,
                    'params':None,
                    'formula':formula,
                    'train_data_set':train_data_set, 
                    'test_data_set':test_data_set,
                    'figure':fig_modele_mesure
                    }

    return model_result



def Run_LGBM(data:pd.DataFrame, split_test_train:bool, params:dict):


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


    
    params['metric'] = 'rmse'
    params['random_state'] = 48

    hp_param_set = {
        f"lgbm__{key}": value for key, value in params.items()
    }

    model.set_params(**hp_param_set)


    if split_test_train:
        train_data_set, test_data_set = train_test_split(data, test_size=0.33, random_state=42)
        train_x = train_data_set.drop(target,axis=1)
        train_y = train_data_set[target]
        model.fit(train_x,train_y)
    else:
        train_data_set = data.copy()
        test_data_set  = None
        train_x = train_data_set.drop(target,axis=1)
        train_y = train_data_set[target]
        model.fit(train_x,train_y)
       
    R2_train, R2_test, mape_train, mape_test = Compute_Perf_Model(model,train_data_set,test_data_set)


    metrics =   {'R2_train':R2_train,
                 'R2_test':R2_test,
                 'mape_train': mape_train, 
                 'mape_test': mape_test}

    fig_modele_mesure = Plot_ymod_ymes(model,train_data_set,test_data_set)


    model_result = {'model':model,
                    'model_type':'LGBMRegressor',
                    'metrics':metrics,
                    'params':params,
                    'formula':'',
                    'train_data_set':train_data_set, 
                    'test_data_set':test_data_set,
                    'figure':fig_modele_mesure
                    }

    return model_result