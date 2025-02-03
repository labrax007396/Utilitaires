def Features_Selection_RFECV(data, dico_model, algo = 'DecisionTreeRegressor'):

    from sklearn.feature_selection import RFECV
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression

    data.dropna(inplace=True)
    X = data.drop(columns=[dico_model["tag_name"]])
    y = data[dico_model["tag_name"]]

    if algo == 'DecisionTreeRegressor':
        estimator = DecisionTreeRegressor()
    elif algo == 'RandomForestRegressor':
        estimator = RandomForestRegressor()
    elif algo == 'GradientBoostingRegressor':
        estimator = GradientBoostingRegressor()
    elif algo ==  'LinearRegression':
        estimator = LinearRegression()
             
    else:
        print("nom algo incorrect")
        return None
          


    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(X, y)

    selection_features = dict()
    features_name = list(X.columns)

    for f,s,r in zip(features_name,selector.support_,selector.ranking_) :
        selection_features[f] = {'pertinent':s,'ordre':r}

    return selection_features











def Compute_Features_Importance(data):

    from sklearn.tree import DecisionTreeRegressor
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    data_cpy = data.copy(deep=True)
    data_cpy.dropna(inplace=True)

    ipe_name = data_cpy.columns[0]
    fact_name = data_cpy.columns[1:]

    X = data_cpy[fact_name]
    y = data_cpy[ipe_name]
    X = pd.get_dummies(X)
    cols = list(X.columns)
    arbre_preselect = DecisionTreeRegressor()
    arbre_preselect.fit(X = X, y = y)

    df_feat_imp = pd.DataFrame({'Poids Facteur %':100*arbre_preselect.feature_importances_},index=cols)
    df_feat_imp.sort_values(by=['Poids Facteur %'], ascending=False,inplace=True)
    
    #plt.figure(figsize=(8,10))

    #b = sn.barplot(x='Poids Facteur %',y='Facteur', orient="h",data=df_feat_imp)
    #b.set_xlabel("Poids Facteur %", size=16)
    #b.set_ylabel("Facteur", size=16)

    return df_feat_imp

def Compute_Corr_Coef(data = None,dico_model = None):

    import pandas as pd
    var_numerique = [v for v in data.columns if data[v].dtypes != 'object']
    data_num = data[var_numerique]
    df_num_corr = data_num.corr().drop(dico_model['tag_name'])[dico_model['tag_name']] 
    df_num_corr = df_num_corr.to_dict()

    var_cat  = [v for v in data.columns if data[v].dtypes == 'object']

    for vc in var_cat:

        data_cat = data[[dico_model['tag_name']]+[vc]]
        data_cat = pd.get_dummies(data_cat)
        df_cat_corr = data_cat.corr().drop(dico_model['tag_name'])[dico_model['tag_name']] 
        df_num_corr[vc] = df_cat_corr.max()

    return df_num_corr


def Features_Selection(data = None, dico_model = None):

    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    import pandas as pd
    var_numerique = [v for v in data.columns if data[v].dtypes != 'object']
    data_num = data[var_numerique]

    y_train = data_num[dico_model['tag_name']]
    X_train = data_num.drop(dico_model['tag_name'],axis=1)
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)


    # configure to select all features
    fs_mut = SelectKBest(score_func=mutual_info_regression, k='all')
    fs_mut.fit(X_train, y_train)

    scores     = dict()
    scores_mut = dict()

    p_values   = dict()

    for feat, score, p, score_m in zip(X_train.columns, fs.scores_, fs.pvalues_, fs_mut.scores_):
        scores[feat]   = score
        p_values[feat] = p
        scores_mut[feat]   = score_m

    coef_corr = Compute_Corr_Coef(data = data, dico_model = dico_model)

    scores = pd.Series(scores)
    p_values = pd.Series(p_values)
    coef_corr = pd.Series(coef_corr)
    scores_mut = pd.Series(scores_mut)

    df_feat_select      = pd.concat([scores,scores_mut,p_values,coef_corr],axis=1)
    df_feat_select.columns = ['F_score','F_mutual','P_value','Corr_Coef']

    return df_feat_select

def Plot_ymod_ymes(y_train, y_train_modelise,y_test, y_test_modelise):
    import matplotlib.pyplot as plt
    #%matplotlib inline
    plt.figure(figsize=(8,8))
    plt.scatter(y_train, y_train_modelise, marker= 'o', s=30, alpha=0.8,label='Apprentissage')

    if y_test is not None:
        plt.scatter(y_test, y_test_modelise, marker= 's',color='green', s=30, alpha=0.8,label='Validation')
    plt.plot([0,y_train.max()], [0,y_train.max()], 'r-')
    
    plt.xlabel('mesuré')
    plt.ylabel('calculé')
    plt.axis('equal')
    plt.axis([0,y_train.max(), 0, y_train.max()])
    plt.legend()
    plt.show()
        








