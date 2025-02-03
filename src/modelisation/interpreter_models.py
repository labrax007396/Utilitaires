import numpy as np
import scipy
import pandas as pd
from mlflow.tracking import MlflowClient
import shap
import sklearn, lightgbm
import mlflow
import os
import sys
import importlib
import commentjson
import warnings

# Librairies personnelles
# = os.getcwd()
#path_src = os.path.abspath(os.path.join(path, os.pardir,"../src"))
#sys.path.append(path_src)
#path_mode = os.path.abspath(os.path.join(path, os.pardir,"src","modelisation"))
#sys.path.append(path_mode)

from modelisation import mlflow_functions
importlib.reload(mlflow_functions)

from modelisation import build_run_models
from utilitaires import utilitaires

from modelisation import rapport_modelisation as modelreport
from modelisation import convert_modele as conv_mod
    

ALLOWED_MODEL_CLASS = [lightgbm.sklearn.LGBMRegressor,sklearn.ensemble.GradientBoostingRegressor,sklearn.ensemble.RandomForestRegressor]

def create_shapeley_onnx_models(run:mlflow.entities.run.Run, int_config:dict):

    # on vérifie que tous les facteurs sont dans le dataframe shape values

    try:
        shape_data_file = mlflow.artifacts.download_artifacts(run_id=run.info.run_id,artifact_path="interprétation/shape.csv")
        shape_data = pd.read_csv(shape_data_file)
        if 'facteurs' in int_config.keys():
            for f in int_config['facteurs']:
                if f not in shape_data.columns:
                    print("Shape value non calculée pour: " + f)
                    return
            facteurs = int_config['facteurs']
        else:
            facteurs = shape_data.columns.to_list()
            facteurs.remove('Date')


     
    except OSError:
        print("shape values non encore calculées")
        return



    for feature in facteurs:
        create_shapeley_onnx_model(run=run, facteur=feature, int_config=int_config)




def create_shapeley_onnx_model(run:mlflow.entities.run.Run, facteur:str, int_config:dict):

    ''' Permet de créer le modèle onnx pour 1 seul facteur à partir du run d'une expérience
    inputs: 
            * run : run de l'expérience mlflow
            * facteur : nom du facteur 
            * int_config: dictionnaire de paramétrage de l'intérprétation
    
    
    '''


    ## On récupère les données shape de chaque facteur

    import datetime
    date_parser = lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S")

    try:
        shape_data_file = mlflow.artifacts.download_artifacts(run_id=run.info.run_id,artifact_path="interprétation/shape.csv")

        shape_data = pd.read_csv(shape_data_file)
        shape_data["Date"] = shape_data["Date"].apply(date_parser)
        shape_data.set_index('Date',inplace=True)


    except OSError:
        print("shape values non encore calculées")
        return
    
    
    if facteur not in shape_data.columns:
        print("Le facteur " + facteur + " est absent")
        return
    
    df_shape_fact = shape_data[facteur]
    df_shape_fact.name = facteur+'_shape'

    ## On récupère les données du run stockées dans le repository

    dico_model_data = mlflow_functions.Load_Information_From_run(run)

    ## On remplace la 1ère colonne par les données shape du facteur

    ipe = dico_model_data['data'].columns[0]
    dico_model_data['data'].drop(columns=[ipe],inplace=True)
    dico_model_data['data'] = pd.concat([df_shape_fact,dico_model_data['data']],axis=1)

    # On récupère les hyper paramètres du modèle

    dico_run = dict(run)
    params = dict(dico_run['data'])['params']

    hp_params = dict()

    for k,v in params.items():
        if type(v) in [int ,float]:
            hp_params[k] = v
        else:  
            if v.isdigit(): #entier
                hp_params[k] = int(v)
            else:
                try:
                    hp_params[k] = float(v)
                except:
                    hp_params[k] = v

    # On modifie le fichier de configuration du modèle pour le configurer à la shape value

    dico_model_data['model_config']['mangling'] = int_config["mangling_interp"]
    dico_model_data['model_config']['tag_name'] = facteur+"_shape"
    dico_model_data['model_config']['description'] = dico_model_data['model_config']['description'] + ' ' + facteur + ' shape value'

    # calcul des coefficients de corrélation pour les stocker dans le rapport modélisation du modele onnx

    df_num_corr = utilitaires.Compute_Corr_Coef(data=dico_model_data['data'], dico_model =dico_model_data['model_config'])

    for t, df in dico_model_data["model_config"]["facteurs"].items():
        if not df['used']:
            df_num_corr[df['nom']] = 0.0

    # On apprend le modèle

    print("Apprentissage du modèle de: "+ facteur + ' shape value')

    model_result = build_run_models.Run_LGBM(data=dico_model_data['data'], split_test_train=True, params=hp_params)
    
    print(f'R2_train: {model_result["metrics"]["R2_train"]:.3f} R2_test: {model_result["metrics"]["R2_test"]:.3f}')

    # On récupère le rapport modélisation du modèle de l'IPE pour l'adapter

    model_repport_file = mlflow.artifacts.download_artifacts(run_id=run.info.run_id,artifact_path="Rapports/rapport_modelisation.json")
    with open(model_repport_file, encoding='utf-8') as file:
        model_repport= commentjson.load(file)

    clean_report = dict()
    clean_report['ndata_before_clean'] = model_repport["dataframe_info"]["cleaning_info"]["line_count_before"]
    clean_report['ndata_after_clean'] =  model_repport["dataframe_info"]["cleaning_info"]["line_count_after"]
    clean_report['detail'] = dict()
    clean_report['detail']['target'] = dict()
    clean_report['detail']['target']['missingdata'] = model_repport["dataframe_info"]["target"]["continous_serie_info"]["missingdata"]
    clean_report['detail']['target']['outliers'] = model_repport["dataframe_info"]["target"]["continous_serie_info"]["outliers"]
    clean_report['detail']['features'] = dict()
    f_dic = model_repport['dataframe_info']['features']
    for fd in f_dic:
        t = 'tag_'+str(fd['tag_id'])
        clean_report['detail']['features'][t] = dict()
        if fd['continous_serie_info'] is not None:
            clean_report['detail']['features'][t]['missingdata'] = fd['continous_serie_info']['missingdata']
            clean_report['detail']['features'][t]['outliers'] = fd['continous_serie_info']['outliers']

    # Réalisation du rapport modélisation

    modelreport_json = modelreport.BuildModelReport(model_type  = model_result['model_type'],
                                            ref_periode_debut  = datetime.datetime.strftime(dico_model_data['data'].index[0], '%Y-%m-%d %H:%M:%S')  ,
                                            ref_periode_fin= datetime.datetime.strftime(dico_model_data['data'].index[-1], '%Y-%m-%d %H:%M:%S'),
                                            clean_report = clean_report,
                                            description = '',
                                            formula=model_result['formula'],
                                            test_data_set = model_result['test_data_set'],
                                            train_data_set = model_result['train_data_set'],
                                            fitted_model = model_result['model'],
                                            df_num_corr = df_num_corr,
                                            dico_model = dico_model_data['model_config'],
                                            data = dico_model_data['data'])
    # Conversion du modèle au format onnx

    target = model_result['train_data_set'].columns[0]
    train_x = model_result['train_data_set'].drop(target,axis=1)


    model_onnx = conv_mod.convert_to_onnx(train_x, model_result['model'], modelreport_json)


    onnx_model_name = facteur+'_shape_test.onnx'
        
    with open(int_config["tmp_file"]+onnx_model_name, "wb") as f:
            f.write(model_onnx.SerializeToString()) 

    with mlflow.start_run(run_id=run.info.run_id) as run:
        mlflow.log_artifact(int_config["tmp_file"]+onnx_model_name, artifact_path="interprétation")

    mlflow.end_run()     

    # Suppression des fichiers temporaires

    for f in os.listdir(int_config["tmp_file"]):
        os.remove(os.path.join(int_config["tmp_file"], f))



def Compute_Shape_Values(dico_model_data:dict):

    target = dico_model_data['data'].columns[0]
    train_x = dico_model_data['data'].drop(target,axis=1)
    train_y = dico_model_data['data'][target]

    if type(dico_model_data['model']) == sklearn.pipeline.Pipeline: # Le modèle est un Pipeline on transforme les entrées avant d'interpréter

        x_transformed = dico_model_data['model'].steps[0][1].transform(train_x)
        if type(x_transformed) == scipy.sparse._csr.csr_matrix:
            x_transformed = np.asarray(x_transformed.todense())


        f_name = dico_model_data['model'].steps[0][1].get_feature_names_out()
        f_name = [f.replace('num__','').replace('cat__','') for f in f_name]

        mapping_alias_desc = {f['nom']:f['description'] for t, f in dico_model_data['model_config']['facteurs'].items()}
        mapping_alias_type = {f['nom']:f['type'] for t, f in dico_model_data['model_config']['facteurs'].items()}

        new_label = list()

        for f in f_name:

            for a in list(mapping_alias_desc.keys()):

                # si variables catégorielle
                if mapping_alias_type[a] == 'cat':
                    if a in f:
                        nl = f.replace(a, mapping_alias_desc[a])
                        new_label.append(nl)   

                elif mapping_alias_type[a] == 'num':
                    if a==f:
                        nl = f.replace(a, mapping_alias_desc[a])
                        new_label.append(nl)

        df_x = pd.DataFrame(index=train_x.index,data=x_transformed,columns=new_label)

        model = dico_model_data['model'].steps[1][1]

    else:
        df_x = train_x.copy()
        model = dico_model_data['model']


    if model.__class__ in ALLOWED_MODEL_CLASS:

        explainer = shap.TreeExplainer(dico_model_data['model'].steps[1][1], 
                                    model_output='raw', 
                                    feature_perturbation='interventional' 
                                    )
    
    else:
        print("type de modèle non pris en charge")
        return None
    

    svals = explainer.shap_values(df_x, y=train_y)
    #df_svals = pd.DataFrame(index = df_x.index,data=svals,columns=f_name)


    return svals, df_x,f_name



def plot_graphes(svals, df_x):

    import matplotlib.pyplot as plt
    dico_graphes = dict()

    fig_importance_fact = plt.figure()
    ax_fi = fig_importance_fact.axes
    shap.summary_plot(svals, df_x, plot_type="bar",show=False)
    plt.gca().tick_params(labelsize=10)
    plt.gca().set_xlabel("Impact moyen sur la variable modélisée", fontsize=14)


    fig_shape_fact = plt.figure()
    ax_shape = fig_shape_fact.axes
    shap.summary_plot(svals, df_x,show=False)
    plt.gca().tick_params(labelsize=10)
    plt.gca().set_xlabel("Impact sur la variable modélisée", fontsize=14)


    dico_graphes["fig_importance_fact"] = fig_importance_fact
    dico_graphes["fig_shape_fact"]      = fig_shape_fact


    return dico_graphes