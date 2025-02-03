
import pandas as pd
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures,MinMaxScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,ElasticNet,Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
from sklearn.base import BaseEstimator

import numpy as np


#def Run_Model(type:str, data:pd.DataFrame, split_test_train:bool, params:dict):



def Build_And_Train_Model(data:pd.DataFrame, model_type:str, split_test_train:bool, params:dict, dico_model:dict):


    if model_type == "LinearRegression":
        preproc        = Build_Preproc(data,dico_model)
        model          = Pipeline([("preprocessor",preproc),("model", LinearRegression())])
        model_result   = Run_Model(model,data,split_test_train,model_type,params,dico_model)

    if model_type == "PolyRegression":
        preproc = Build_Preproc_Poly(data=data, dico_model=dico_model, deg=params['degre'],interaction_only=params['interaction_only'],normalize=True)
        model          = Pipeline([("preprocessor",preproc),("model", Ridge())])
        hp_param_set = Train_Ridge_Regression(data,model)
        model.set_params(**hp_param_set)
        model_result   = Run_Model(model,data,split_test_train,model_type,params,dico_model)

    if model_type == "Powregression":
        model = PowerRegression(params['a'],params['b'],params['c'])
        model_result   = Run_Model(model,data,split_test_train,model_type,params,dico_model)


    if model_type == "ElasticNet":
        preproc        = Build_Preproc(data,dico_model)
        model          = Pipeline([("preprocessor",preproc),("model", ElasticNet())])
        hp_param_set = {
            f"model__{key}": value for key, value in params.items()
        }
        model.set_params(**hp_param_set)
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
        model_lower, model_upper = create_upper_lower_models(model,data)
        model_result['model_upper'] = model_upper
        model_result['model_lower'] = model_lower


    if model_type == "DecisionTreeRegressor":
        preproc        = Build_Preproc(data,dico_model)
        model          = Pipeline([("preprocessor",preproc),("model", DecisionTreeRegressor())])
        hp_param_set = {
            f"model__{key}": value for key, value in params.items()
        }
        model.set_params(**hp_param_set)
        model_result   = Run_Model(model,data,split_test_train,model_type,params,dico_model)


    return model_result


def Build_Preproc(data:pd.DataFrame, dico_model:dict):

    import warnings
    warnings.simplefilter('ignore')

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



def Build_Preproc_Poly(data:pd.DataFrame, dico_model:dict, deg:int,interaction_only:bool,normalize:bool):
    
    
    num_feat = [f for f in data.columns[1:] if data.dtypes[f]==np.float64]
    cat_feat = [f for f in data.columns[1:] if data.dtypes[f]==object]

    if normalize:
        num_transformer   = Pipeline(steps=[('minmax', MinMaxScaler()),('poly',PolynomialFeatures(degree=deg,interaction_only=interaction_only))])
    else:
        num_transformer   = Pipeline(steps=[('poly',PolynomialFeatures(degree=int,interaction_only=interaction_only))])


    
    if len(cat_feat)>0:

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

            cat_transformer = Pipeline(steps=[('cat', OneHotEncoder(categories=list_categories,handle_unknown='ignore'))])

        else:
            cat_transformer = Pipeline(steps=[('cat', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
                remainder='passthrough', #passthough features not listed
                transformers=[
                    ('num', num_transformer , num_feat),
                    ('cat', cat_transformer , cat_feat)
                ])

    else:
        preprocessor = ColumnTransformer(remainder='passthrough', transformers=[('num', num_transformer , num_feat)])

    return preprocessor


def Train_Ridge_Regression(data,pipeline):

    from sklearn.model_selection import GridSearchCV

    data.dropna(inplace=True)

    X=data[data.columns[1:]]
    y=data[data.columns[0]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    search = GridSearchCV(pipeline,
                        {'model__alpha':np.arange(0.1,10,0.5)},
                        cv = 5, scoring="r2",verbose=3
                        )
    search.fit(X_train,y_train)


    return search.best_params_


def Run_Model(model,data,split_test_train,model_type,params,dico_model):

    
    import warnings
    warnings.simplefilter('ignore')


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
        df_pertinence = RegressionWithStatsmodels(data=data,dico_model=dico_model)

    elif model_type == "PolyRegression":
        formula = Formula_RegPoly(model)
        df_pertinence = pd.DataFrame()

    elif model_type == "Powregression":
        formula = model.get_formula()
        df_pertinence = pd.DataFrame()
   
    else:
        formula = ''
        df_pertinence = pd.DataFrame()

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
                    'figure':fig_modele_mesure,
                    'pertinence':df_pertinence
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


    # initializing list
 

def Run_RegLin_All_Combination_Features(data:pd.DataFrame, p_value:float,n_max_feat:int,n_mint_feat:int):

    
    import warnings
    warnings.simplefilter('ignore')


    import numpy as np
    import statsmodels.formula.api as smf
    from itertools import chain, combinations

    features_list = data.columns[1:].to_list()
    feat_comb = []
    for sub in range(n_mint_feat-1,n_max_feat):
        feat_comb.extend(combinations(features_list, sub + 1))


    df_models = pd.DataFrame(columns=['Features','R2'])
    iter_model = 0


    for subset in feat_comb:
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

def RegressionWithStatsmodels(data:pd.DataFrame, dico_model:dict):

    import statsmodels.formula.api as smf
    import pandas as pd

    data.dropna(inplace=True)
    formula = data.columns[0] + '~'
    formula = formula + '+'.join(list(data.columns[1:]))
    model_smf = smf.ols(formula=formula, data=data).fit()

    pvalues   = model_smf.pvalues.tolist()
    list_fact = model_smf.pvalues.index.tolist()


    dico_facteurs = dico_model["facteurs"]
    list_nom  = [d['nom'] for d in dico_facteurs.values() if d['used']]
    list_desc = [d['description'] for d in dico_facteurs.values() if d['used']]

    for nom, desc in zip(list_nom, list_desc):
        list_fact = [l.replace(nom,desc) for l in list_fact]

    df_pertinence = pd.DataFrame(data={'Facteur':list_fact,'Pvalue':pvalues})
    df_pertinence = df_pertinence.sort_values(by='Pvalue')
    df_pertinence['Pertinent'] = df_pertinence.apply(lambda row: 'Oui' if row['Pvalue']<=0.05 else 'Non',axis=1)

    return df_pertinence











def get_coef_pvalues(train_X,train_y,model):
    
    import numpy as np
    from scipy.stats import t

    X = train_X.values
    y = train_y.values
    
    one = np.ones((len(X))).reshape(len(X),1)
    X = np.hstack((one,X))
    # Calculate the residuals
    coefficients = np.insert(model.coef_,0,model.intercept_)
    y_pred = X @ coefficients
    residuals = y - y_pred


    # Calculate the residual sum of squares (RSS)
    RSS = np.sum(residuals ** 2)


    # Calculate the degrees of freedom
    n = X.shape[0]
    p = X.shape[1] - 1
    df = n - p - 1


    # Calculate the standard error of the coefficients
    XTX_inv = np.linalg.inv(X.T @ X)
    coef_se = np.sqrt(np.diagonal(XTX_inv) * RSS / df)
    # Calculate the t-statistic and p-value for each coefficient
    t_stat = coefficients / coef_se
    p_values = (1 - t.cdf(np.abs(t_stat), df)) * 2

    return p_values




def Formula_RegPoly(model):

    formula = str(model.named_steps['model'].intercept_)
    coefficients = model.named_steps['model'].coef_
    features = model.named_steps['preprocessor'].get_feature_names_out()

    for coef, feat in zip(coefficients, features):
        if np.abs(coef) > 0.000001:
            feat_renamed = feat.replace('num__','').replace('cat__','').replace(' ','*')
            if coef>0:
                sign='+'
            else:
                sign='-'
            formula = formula + sign + str(np.abs(coef)) + "*" + feat_renamed
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


def create_upper_lower_models(model,data):
    coltransf = model.steps[0][1]
    best_params = model.steps[1][1].get_params()

    best_params['objective'] = 'quantile'
    best_params['alpha'] = 1 - 0.95

    est_lower = LGBMRegressor()
    est_lower.set_params(**best_params)
    model_lower = Pipeline([("preprocessor",coltransf),("model", est_lower)])


    best_params['alpha'] = 0.95

    est_upper = LGBMRegressor()
    est_upper.set_params(**best_params)
    model_upper = Pipeline([("preprocessor",coltransf),("model", est_upper)])

    data.dropna(inplace=True)
    target = data.columns[0]   
    X = data.drop(target,axis=1)
    y = data[target]


    model_lower.fit(X,y)
    model_upper.fit(X,y)
    return model_lower, model_upper


class PowerRegression(BaseEstimator):

    ''' Fittage regression exponentielle y = a*x**b+c '''

    def __init__(self, a,b,c):
        self.a_init  = a
        self.b_init  = b
        self.c_init  = c

    def fit(self, X, y):
       
        import warnings
        warnings.simplefilter('ignore')
        from scipy.optimize import curve_fit

        p_init = [self.a_init,self.b_init,self.c_init]

        X = X[X.columns[0]]

        popt, _ = curve_fit(self.objective, X.values,y.values,p0=p_init)
        self.a, self.b, self.c = popt

        self.X_ = X
        self.y_ = y

        #return self
    

    # objective function
    def objective(self,x, a, b,c):
        return a * x**b +c



    def predict(self, X):
        X = X[X.columns[0]]
        ypred = self.objective(X.values, self.a, self.b, self.c)
        return ypred

    def score(self,X,y):
        from sklearn.metrics import r2_score
        import pandas as pd
        ypred = self.predict(X)

        return r2_score(y.values,ypred)
    
    def get_formula(self):
        formula = str(self.a)+'*[' + self.X_.name + '].Pow(' + str(self.b) + ')+(' + str(self.c) + ')'
        return formula


    def convert_to_onnx(self):


        from onnx import numpy_helper, TensorProto
        from onnx.helper import (
        make_model, make_node,
        make_graph, make_tensor_value_info,make_operatorsetid)
        from onnxruntime import (__version__ as ort_version)
        import sys, os

        try:

            # input and output
            X = make_tensor_value_info(
                self.X_.name, TensorProto.FLOAT, [None, 1])
            
            Y = make_tensor_value_info(
                'target', TensorProto.FLOAT, [None, 1])

            # inference
            node_power = make_node('Pow', [self.X_.name, 'b'], ['X_pow'], name='N1')

            node_matmul = make_node('Mul', ['a', 'X_pow'], ['a_X_pow'], name='N2')

            node_add   = make_node('Add', ['a_X_pow', 'c'], ['target'], name='N3')

            # initializer

            a = np.array([self.a]).astype(np.float32)
            c = np.array([self.b]).astype(np.float32)
            b = np.array([self.c]).astype(np.float32)

            init_a = numpy_helper.from_array(a, name="a")
            init_b = numpy_helper.from_array(b, name="b")
            init_c = numpy_helper.from_array(c, name="c")

            # graph
            graph_def = make_graph(
                [node_power, node_matmul, node_add], 'power', [X], [Y],
                [init_a,init_b, init_c])
            
            self.model_onnx = make_model(
                graph_def, producer_name='orttrainer', ir_version=7,
                producer_version=ort_version,
                opset_imports=[make_operatorsetid('', 14)])
            

        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            self.MessageLogger.write_msg('erreur',"Erreur conversion onnx: "+type(error).__name__ +" " +fname+ " Ligne "+ str(exc_tb.tb_lineno))
            self.data =  pd.DataFrame


