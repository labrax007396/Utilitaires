def BuildModelReport(model_type  = None,
                     ref_periode_debut  ="ref_periode_debut"  ,
                     ref_periode_fin="ref_periode_fin",
                     formula = '',
                     clean_report = None,
                     description = None,
                     test_data_set = None,
                     train_data_set = None,
                     fitted_model = None,
                     df_num_corr = None,
                     dico_model = None,
                     data = None):



    from sklearn.metrics import r2_score
    from datetime import datetime
    import pandas as pd
    import numpy as np
    #import keras
    
    ######### Calcul des sensibilité des facteurs #########

    print("Calcul des sensibilités des facteurs")

    dico_fact_sens = dict()
    X = data.drop(columns=dico_model['tag_name'])

    for fact in X.columns:
        if X[fact].dtypes == 'float64':   

            df_25 = pd.DataFrame(columns=X.columns.to_list())
            list_25 = list()
            df_75 = pd.DataFrame(columns=X.columns.to_list())
            list_75 = list()
            for fact1 in X.columns:
                if X[fact1].dtypes == 'float64':
                    list_25.append(X[fact1].mean())
                    list_75.append(X[fact1].mean())
                else:
                    list_25.append('other')
                    list_75.append('other')
            df_25.loc[0] = list_25 
            df_25[fact]  = X[fact].min() + 0.25*(X[fact].max() - X[fact].min())
            df_75.loc[0] = list_75 
            df_75[fact]  = X[fact].min() + 0.75*(X[fact].max() - X[fact].min())    

            if model_type=="rnn":

                features_dict_25 = {name: np.array(value) 
                         for name, value in df_25.items()}

                features_dict_75 = {name: np.array(value) 
                         for name, value in df_75.items()}

                pred_25 = fitted_model.predict(features_dict_25)[0]
                pred_75 = fitted_model.predict(features_dict_75)[0]
                

            else:    

                pred_25 = fitted_model.predict(df_25)
                pred_75 = fitted_model.predict(df_75)

            dico_fact_sens[fact] = pred_75[0] - pred_25[0]

        else:

            list_mod_unique = list(X[fact].unique())



            df_sans_mod = pd.DataFrame(columns=X.columns.to_list())
            list_sans_mod = list()
            df_avec_mod  = pd.DataFrame(columns=X.columns.to_list())

            list_avec_mod = list()

###############################################################################

            list_fact_num = [f for f in X.columns if X[f].dtypes == 'float64']

            dico_sans_mod = dict()

            for fact1 in X.columns:
                if X[fact1].dtypes == 'float64':
                    dico_sans_mod[fact1] = X[fact1].mean()
                else:
                    dico_sans_mod[fact1] = 'other'

            dico_avec_mod = dico_sans_mod.copy()
            df_sans_mod = pd.DataFrame(data = dico_sans_mod,index = [0])

            if model_type=="rnn":

                features_dict_sans_mod = {name: np.array(value) 
                         for name, value in df_sans_mod.items()}

                pred_sans_mod = fitted_model.predict(features_dict_sans_mod)[0]



            else:


                pred_sans_mod = fitted_model.predict(df_sans_mod)

            if model_type=="rnn":

                for mod in list_mod_unique:

                    dico_avec_mod[fact] = mod
                    df_avec_mod = pd.DataFrame(data = dico_avec_mod,index = [0])

                    features_dict_avec_mod = {name: np.array(value) 
                            for name, value in df_avec_mod.items()}

                    pred_avec_mod = fitted_model.predict(features_dict_avec_mod)[0]
                    dico_fact_sens[mod] = pred_avec_mod[0] - pred_sans_mod[0]                


            
            else:

                for mod in list_mod_unique:

                    dico_avec_mod[fact] = mod
                    df_avec_mod = pd.DataFrame(data = dico_avec_mod,index = [0])
                    pred_avec_mod = fitted_model.predict(df_avec_mod)
                    dico_fact_sens[mod] = pred_avec_mod[0] - pred_sans_mod[0]                
                    


            #print(list(dico_avec_mod.values()))
                


            
################################################################################

            #for fact1 in X.columns:
            #    if X[fact1].dtypes == 'float64':
            #        list_sans_mod.append(X[fact1].mean())
            #        list_avec_mod.append(X[fact1].mean())
            #    else:
            #        list_sans_mod.append('other') 
                
            #        df_sans_mod.loc[0] = list_sans_mod
                    
            #        pred_sans_mod = fitted_model.predict(df_sans_mod)
                   

            #        for mod in list_mod_unique:
            #            list_avec_mod.append(mod)
            #            df_avec_mod.loc[0] = list_avec_mod
            #            pred_avec_mod = fitted_model.predict(df_avec_mod)
            #            dico_fact_sens[mod] = pred_avec_mod[0] - pred_sans_mod[0]                
             #           list_avec_mod.remove(mod)


    ######### Calcul des scores du modèle  #########

    #print("Calcul des scores du modèle")

    if test_data_set is not None:
    
        if isinstance(train_data_set, pd.DataFrame):
            test_data = test_data_set.drop(columns=dico_model["tag_name"])
            y_test = test_data_set[dico_model["tag_name"]].to_frame()
            train_data = train_data_set.drop(columns=dico_model["tag_name"])
            y_train = train_data_set[dico_model["tag_name"]].to_frame()
        else:
            y_test    = test_data_set.keep_columns(dico_model['tag_name']).to_pandas_dataframe()
            test_data = test_data_set.drop_columns(dico_model['tag_name']).to_pandas_dataframe()
            y_train    = train_data_set.keep_columns(dico_model['tag_name']).to_pandas_dataframe()
            train_data = train_data_set.drop_columns(dico_model['tag_name']).to_pandas_dataframe()


        if model_type=="rnn":

            #print("Calcul predictions train")
            features_dict_train = {name: np.array(value) for name, value in train_data.items()}
            y_pred_train = fitted_model.predict(features_dict_train).flatten()
            #print("Calcul predictions test")
            features_dict_test = {name: np.array(value) for name, value in test_data.items()}
            y_pred_test  = fitted_model.predict(features_dict_test).flatten()

        else:


            #print("Calcul predictions train")
            y_pred_train = fitted_model.predict(train_data)
            #print("Calcul predictions test")
            y_pred_test  = fitted_model.predict(test_data)

       

        resu_train = pd.DataFrame({'y_pred_train':y_pred_train, 'y_train':y_train[dico_model['tag_name']]})
        resu_test = pd.DataFrame({'y_pred_test':y_pred_test, 'y_test':y_test[dico_model['tag_name']]})

        # calcul des métriques sur la période d'apprentissage

        ndata_train = len(y_pred_train)
        r2_train = r2_score(y_train,y_pred_train)
        
        resu_train = resu_train[resu_train['y_train']>0]
        resu_train['erreur']     = resu_train.apply(lambda row: abs(row['y_pred_train']-row['y_train']),axis=1)
        resu_train['erreur_rel'] = resu_train.apply(lambda row: 100*abs((row['y_pred_train']-row['y_train'])/row['y_train']),axis=1)

        mean_deviation_train = resu_train['erreur'].mean()
        standard_deviation_train = resu_train['erreur'].std()
        mape_train = resu_train['erreur_rel'].mean()



        # calcul des métriques sur la période de test

        ndata_test  = len(y_pred_test)
        r2_test  = r2_score(y_test,y_pred_test)
        

        resu_test = resu_test[resu_test['y_test']>0]
        resu_test['erreur']     = resu_test.apply(lambda row: abs(row['y_pred_test']-row['y_test']),axis=1)
        resu_test['erreur_rel'] = resu_test.apply(lambda row: 100*abs((row['y_pred_test']-row['y_test'])/row['y_test']),axis=1)

        mean_deviation_test = resu_test['erreur'].mean()
        standard_deviation_test = resu_test['erreur'].std()
        mape_test = resu_test['erreur_rel'].mean()




    else: # Pas de données de validation

        if isinstance(train_data_set, pd.DataFrame):
            train_data = train_data_set.drop(columns=dico_model["tag_name"])
            y_train = train_data_set[dico_model["tag_name"]]
        else:
            y_train    = train_data_set.keep_columns(dico_model['tag_name']).to_pandas_dataframe()
            train_data = train_data_set.drop_columns(dico_model['tag_name']).to_pandas_dataframe()



        if model_type=="rnn":

            #print("Calcul predictions train")
            features_dict_train = {name: np.array(value) for name, value in train_data.items()}
            y_pred_train = fitted_model.predict(features_dict_train).flatten()

        else:

            #print("Calcul predictions train")
            y_pred_train = fitted_model.predict(train_data)
        
        resu_train = pd.DataFrame(data = {'y_pred_train':y_pred_train, 'y_train':y_train})


        r2_train = r2_score(y_train,y_pred_train)
        r2_test  = None


        # calcul des métriques sur la période d'apprentissage

        ndata_train = len(y_pred_train)
        r2_train = r2_score(y_train,y_pred_train)
        
        resu_train = resu_train[resu_train['y_train']>0]
        resu_train['erreur']     = resu_train.apply(lambda row: abs(row['y_pred_train']-row['y_train']),axis=1)
        resu_train['erreur_rel'] = resu_train.apply(lambda row: 100*abs((row['y_pred_train']-row['y_train'])/row['y_train']),axis=1)

        mean_deviation_train = resu_train['erreur'].mean()
        standard_deviation_train = resu_train['erreur'].std()
        mape_train = resu_train['erreur_rel'].mean()

        ndata_test  = 0
        r2_test  = None

        mean_deviation_test = None
        standard_deviation_test = None
        mape_test = None


    creation_date = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")


    #creation_date =  datetime.now()
    #formula = ''

    if 'description' in dico_model.keys():
        description = dico_model['description']

    else:
        description = 'non définie'

    model_info = ModelInfo(model_type    = model_type,
                        description   = description, 
                        creation_date = creation_date,
                        formula       = formula,
                        r2_test       = r2_test,
                        r2_train      = r2_train,
                        mape_test          = mape_test,
                        mape_train          = mape_train,
                        mean_deviation_train = mean_deviation_train,
                        mean_deviation_test = mean_deviation_test,
                        standard_deviation_train = standard_deviation_train,
                        standard_deviation_test = standard_deviation_test)

    

    cleaninginfo  = CleaningInfo(line_count_before = clean_report['ndata_before_clean'],
                                 line_count_after  = clean_report['ndata_after_clean'])


    target_continousinfo = ContinousSerieInfo(vmin = data[dico_model['tag_name']].min(),
                                              vmax = data[dico_model['tag_name']].max(),
                                              mean = data[dico_model['tag_name']].mean(),
                                              standard_deviation = data[dico_model['tag_name']].std(),
                                              influence_weight = 0.0,
                                              missingdata = clean_report['detail']['target']['missingdata'],
                                              outliers    = clean_report['detail']['target']['outliers'])

    #print("Peuplement des features")

    listfeatures = []
    for tag in dico_model['facteurs']:
        if dico_model['facteurs'][tag]['used']:
            nom_feature = dico_model['facteurs'][tag]['nom']
            if dico_model['facteurs'][tag]['type'] == 'num':   

                influence_weight = dico_fact_sens[nom_feature]
                corr_coef        = df_num_corr[nom_feature]

                feature_continousinfo = ContinousSerieInfo(vmin = data[nom_feature].min(),
                                                        vmax = data[nom_feature].max(),
                                                        mean = data[nom_feature].mean(),
                                                        standard_deviation = data[nom_feature].std(),
                                                        influence_weight = influence_weight,
                                                        missingdata = clean_report['detail']['features'][tag]['missingdata'],
                                                        outliers = clean_report['detail']['features'][tag]['outliers'])

                
                target = Target(tag_id = int(tag[4:]),
                                name   = nom_feature,
                                description = nom_feature,
                                corr_coef = corr_coef,
                                used = True,
                                discrete_serie_info = None,
                                continous_serie_info = feature_continousinfo)


                listfeatures.append(target)

            if dico_model['facteurs'][tag]['type'] == 'cat':

                #features_weight['Facteur'] = features_weight['Facteur'].apply(lambda s:s.replace(nom_feature+'_',''))
                #features_weight.set_index('Facteur',inplace=True)

                cat_var_list = []
            
                cat_count = data[nom_feature].value_counts()
                for code, nbre in cat_count.items():

                    #importance_percent = features_weight.loc[code,'Poids Facteur %']
                    influence_weight = dico_fact_sens[code]

                    cat_var_obj = CategoricalVariable(name=code,
                                                      occurrences=nbre,
                                                      influence_weight = influence_weight)
                    cat_var_list.append(cat_var_obj)

                corr_coef        = df_num_corr[nom_feature]
                disret_info_serie_objet = DiscreteSerieInfo(categorical_variables = cat_var_list)

                target = Target(tag_id = int(tag[4:]),
                                name   = nom_feature,
                                description = nom_feature,
                                corr_coef = corr_coef,
                                used = True,
                                discrete_serie_info = disret_info_serie_objet,
                                continous_serie_info = None)

                listfeatures.append(target)

        else:  ## La feature n'est pas utilisée

            nom_feature = dico_model['facteurs'][tag]['nom']

            if dico_model['facteurs'][tag]['type'] == 'num':   

                influence_weight = 0.0
                corr_coef        = df_num_corr[nom_feature]

                feature_continousinfo = ContinousSerieInfo(vmin = 0.0,
                                                        vmax = 0.0,
                                                        mean = 0.0,
                                                        standard_deviation = 0.0,
                                                        influence_weight = 0.0,
                                                        missingdata = 0,
                                                        outliers = 0)

                target = Target(tag_id = int(tag[4:]),
                                name   = nom_feature,
                                description = nom_feature,
                                corr_coef = corr_coef,
                                used = False,
                                discrete_serie_info = None,
                                continous_serie_info = None)


                listfeatures.append(target)

            if dico_model['facteurs'][tag]['type'] == 'cat':

                influence_weight = 0.0
                corr_coef        = df_num_corr[nom_feature]

                target = Target(tag_id = int(tag[4:]),
                                name   = nom_feature,
                                description = nom_feature,
                                corr_coef = corr_coef,
                                used = False,
                                discrete_serie_info = None,
                                continous_serie_info = None)

                listfeatures.append(target)






    target_modelise = Target(tag_id = int(dico_model['tag_modelise'][4:]),
                name   = dico_model['tag_name'],
                description = dico_model['tag_name'],
                corr_coef = 1.0,
                used = True,
                discrete_serie_info = None,
                continous_serie_info = target_continousinfo)


    dataframeinfo_obj = DataframeInfo(start_date = ref_periode_debut,
                                      end_date   = ref_periode_fin,
                                      cleaning_info = cleaninginfo,
                                      target = target_modelise,
                                      features = listfeatures)

    #weigth_variables_serie_obj = list()

    #for index, row in features_weight.iterrows():
    #    weightvar_obj = WeightVariable(name = row['Facteur'], weight = row['Poids Facteur %'])
    #    weigth_variables_serie_obj.append(weightvar_obj)



    uv_formula_obj =  UVFormula(dico_model = dico_model, data = data)

    modelreport_obj = ReportModel(site = dico_model['site'],
                                  dataframe_info = dataframeinfo_obj,
                                  model_info = model_info,
                                  uv_formula = uv_formula_obj)



    return modelreport_obj.toJson()



    



from typing import List, Optional
from datetime import datetime


class CleaningInfo:
    line_count_before: int
    line_count_after: int

    def __init__(self, line_count_before: int, line_count_after: int) -> None:
        self.line_count_before = line_count_before
        self.line_count_after = line_count_after
    def getdico(self):
        return {"line_count_before":self.line_count_before,"line_count_after":self.line_count_after}


class ContinousSerieInfo:
    vmin: float
    vmax: float
    mean: float
    standard_deviation: float
    influence_weight: float
    missingdata: int
    outliers: int


    def __init__(self, vmin: float, vmax: float, mean: float, standard_deviation: float, influence_weight: float, missingdata: int, outliers:int) -> None:
        self.min = vmin
        self.max = vmax
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.influence_weight = influence_weight
        self.missingdata = missingdata
        self.outliers = outliers

    def toJson(self):
        import json
        return json.dumps(self, default=lambda o: o.__dict__)



class CategoricalVariable:
    name: str
    occurrences: int
    importance_percent: float

    def __init__(self, name: str, occurrences: int, influence_weight: float) -> None:
        self.name = name
        self.occurrences = occurrences
        self.influence_weight = influence_weight

    def getdico(self):
        return {"name": self.name, "occurrences":self.occurrences, "influence_weight":self.influence_weight}


class DiscreteSerieInfo:
    categorical_variables: List[CategoricalVariable]

    def __init__(self, categorical_variables: List[CategoricalVariable]) -> None:
        self.categorical_variables = categorical_variables

    def getlist(self):
        list_ = list()

        for modalite in self.categorical_variables:
            list_.append(modalite.getdico())
        return list_



class WeightVariable:
    name: str
    weigth: float


    def __init__(self, name: str, weight: float) -> None:
        self.name = name
        self.weight = weight


class WeightVariableSerieInfo:
    weigth_variables: List[WeightVariable]

    def __init__(self, weigth_variables: List[WeightVariable]) -> None:
        self.weigth_variables = weigth_variables








class Target:
    tag_id: int
    name: str
    description: str
    corr_coef: float
    used: bool
    discrete_serie_info: Optional[DiscreteSerieInfo]
    continous_serie_info: Optional[ContinousSerieInfo]

    def __init__(self, tag_id: int, name: str, description: str, corr_coef: float, used:bool, discrete_serie_info:Optional[DiscreteSerieInfo], continous_serie_info: Optional[ContinousSerieInfo]) -> None:
        self.tag_id = tag_id
        self.name = name
        self.description = description
        self.corr_coef = corr_coef
        self.used = used
        self.discrete_serie_info = discrete_serie_info
        self.continous_serie_info = continous_serie_info

    def getdico(self):
        dict_ = dict()
        dict_["tag_id"] = self.tag_id
        dict_["name"]   = self.name
        dict_["description"] = self.description
        dict_["corr_coeff"] = self.corr_coef
        dict_["used"] = self.used

        if self.discrete_serie_info is None:
            dict_["discrete_serie_info"] = None
        else:
            dict_["discrete_serie_info"] = dict()
            dict_["discrete_serie_info"]["categorical_variables"] = self.discrete_serie_info.getlist()
            #dict_["discrete_serie_info"] = self.discrete_serie_info.getlist()

        if self.continous_serie_info is None:
            dict_["continous_serie_info"] = None
        else:
            dict_["continous_serie_info"] = self.continous_serie_info.__dict__

        return dict_


    #def toJson(self):
    #    import json
    #    return json.dumps(self, default=lambda o: o.__dict__)


class DataframeInfo:
    start_date: str
    end_date: str
    cleaning_info: CleaningInfo
    target: Target
    features: List[Target]

    def __init__(self, start_date: str, end_date: str, cleaning_info: CleaningInfo, target: Target, features: List[Target]) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.cleaning_info = cleaning_info
        self.target = target
        self.features = features

    def getdico(self):
        dict_ = dict()
        dict_["start_date"] = self.start_date
        dict_["end_date"]   = self.end_date
        dict_["cleaning_info"] = self.cleaning_info.getdico()

        dict_["target"] = self.target.__dict__
        dict_["target"]["continous_serie_info"] = self.target.continous_serie_info.__dict__

        dict_["features"] = list()

        for feat in self.features:
            dict_["features"].append(feat.getdico())



        return dict_


    def toJson(self):
        import json
        return json.dumps(self, default=lambda o: o.__dict__)


class ModelInfo:
    model_type: str
    description: str
    creation_date: datetime
    formula: str
    r2_test: float
    r2_train: float
    mape_train: float
    mape_test: float
    mean_deviation_train: float
    mean_deviation_test: float
    standard_deviation_train: float
    standard_deviation_test: float

    def __init__(self, model_type: str, description: str, creation_date: str, formula: str, r2_test: float, r2_train: float, mape_train: float, mape_test: float, mean_deviation_train: float, mean_deviation_test: float, standard_deviation_train: float, standard_deviation_test: float) -> None:

        self.model_type = model_type
        self.description = description
        self.creation_date = creation_date
        self.formula = formula
        self.r2_test = r2_test
        self.r2_train = r2_train
        self.mape_train = mape_train
        self.mean_deviation_train = mean_deviation_train
        self.standard_deviation_train = standard_deviation_train
        self.mape_test = mape_test
        self.mean_deviation_test = mean_deviation_test
        self.standard_deviation_test = standard_deviation_test        

    def toJson(self):
        import json
        return json.dumps(self, default=lambda o: o.__dict__)


class UVFormula:
    #from typing import TypeVar
    #PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
    import pandas 
    formula: str

    def __init__(self, dico_model: dict, data: pandas.core.frame.DataFrame) -> None:

        #formula = '[model] '

        #for tag in dico_model['facteurs'].keys():
        #    if dico_model['facteurs'][tag]['used']:
        #        formula = formula + " .Arg(" + '"' + dico_model['facteurs'][tag]['nom'] +'"'+ ", [" + tag + "])"

        #formula = formula + " .Outputs(" + '"' + 'variable_out1' + '"' + ")"
        #self.formula = formula


        formula = '[model] '

        for tag in dico_model['facteurs'].keys():
            if dico_model['facteurs'][tag]['used']:
                if dico_model['facteurs'][tag]['type'] == 'num':
                    nom_feat = dico_model['facteurs'][tag]['nom']
                    min_val = str(data[nom_feat].min())
                    max_val = str(data[nom_feat].max())
                    formula = formula + " .Arg(" + '"' + nom_feat +'"'+ ", [" + tag + "]"
                    formula = formula + ", " + min_val + ", " + max_val +  ")"

                elif dico_model['facteurs'][tag]['type'] == 'cat':
                    nom_feat = dico_model['facteurs'][tag]['nom']
                    mod_liste = list(data[nom_feat].unique())
                    mod_liste = '","'.join(map(str,mod_liste))
                    mod_liste = '"'+mod_liste+'"'
                    formula = formula + " .Arg(" + '"' + nom_feat +'"'+ ", [" + tag + "]"
                    formula = formula + ", " +mod_liste +  ")"


        formula = formula + " .Outputs(" + '"' + 'target' + '"' + ")"
        self.formula = formula








class ReportModel:
    site: str
    dataframe_info: DataframeInfo
    model_info: ModelInfo
    uv_formula: UVFormula

    def __init__(self, site: str, dataframe_info: DataframeInfo, model_info: ModelInfo, uv_formula: UVFormula) -> None:
        self.site = site
        self.dataframe_info = dataframe_info
        self.model_info = model_info
        self.uv_formula = uv_formula
        #return json.dumps(self, default=lambda o: o.__dict__)

    

    def myconverter(self,obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        #elif isinstance(obj, datetime.datetime):
        #    return obj.__str__()

        


    def toJson(self):
        import json
        rapport = dict()
        rapport["site"] = self.site
        rapport["dataframe_info"] = self.dataframe_info.getdico()
        rapport["model_info"] = self.model_info.__dict__
        rapport["uv_formula"] = self.uv_formula.__dict__
        
        return json.dumps(rapport, default=self.myconverter) 
