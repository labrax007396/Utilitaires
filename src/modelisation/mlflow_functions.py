
from pretty_html_table import build_table
import pandas as pd
import datetime
import os
import json
import importlib

import modelisation.build_run_models as mdl
import modelisation.rapport_modelisation as modelreport
import modelisation.convert_modele as conv_mod

from onnx.onnx_pb import StringStringEntryProto

importlib.reload(conv_mod)
importlib.reload(mdl)
importlib.reload(modelreport)

from mlflow.models.signature import infer_signature


def Run_Experiment(mlflow=None, 
                   experiment_id="",
                   data=None,
                   split_test_train=True,
                   model_type = "LinearRegression",
                   params=dict(),
                   dico_figure=dict(),
                   dico_model=dict(),
                   clean_report=dict(),
                   df_num_corr=dict(),
                   onnx_model_name=""
                   ):

    
        
    with mlflow.start_run(experiment_id = experiment_id):
            
            print('Lancement apprentissage')

            Log_Stat_Desc(dico_model, mlflow, data)
   
            model_result =  mdl.Build_And_Train_Model(data=data, model_type=model_type, split_test_train=split_test_train, params=params,dico_model=dico_model)

            # Logging des resultats

            print('Logging des resultats')
            dico_figure['Modèle en fonction mesure'] = model_result['figure']



            Log_Figures(dico_figure, mlflow)
            Log_Metrics(model_result['metrics'],mlflow)
            Log_Data(model_result, mlflow)
            mlflow.log_dict(dico_model,"model_config.json")
            Log_Facteur(dico_model, mlflow)
            Log_Model_Summary(model_result, mlflow)
            Log_Params(params, mlflow)


            print('Logging du modèle au format pkl')
            # Logging du modèle au format pkl
            target = model_result['train_data_set'].columns[0]
            train_x = model_result['train_data_set'].drop(target,axis=1)        
            signature = infer_signature(train_x, model_result['model'].predict(train_x))
            mlflow.sklearn.log_model(model_result['model'], "model", signature=signature)

            # Logging du modèle au format onnx
            print('Logging du modèle au format onnx')

            modelreport_json = modelreport.BuildModelReport(model_type  = model_result['model_type'],
                                                    ref_periode_debut  = datetime.datetime.strftime(data.index[0], '%Y-%m-%d %H:%M:%S')  ,
                                                    ref_periode_fin= datetime.datetime.strftime(data.index[-1], '%Y-%m-%d %H:%M:%S'),
                                                    clean_report = clean_report,
                                                    description = '',
                                                    formula=model_result['formula'],
                                                    test_data_set = model_result['test_data_set'],
                                                    train_data_set = model_result['train_data_set'],
                                                    fitted_model = model_result['model'],
                                                    df_num_corr = df_num_corr,
                                                    dico_model = dico_model,
                                                    data = data)

            

            if model_type == "Powregression":
                print(model_type)

                model_result['model'].convert_to_onnx()
                model_onnx = model_result['model'].model_onnx
                model_onnx.metadata_props.append(StringStringEntryProto(key="ReportModel", value = modelreport_json))

            if model_type == "LGBMRegressor":

                model_onnx       = conv_mod.convert_to_onnx(train_x, model_result['model'], modelreport_json)
                model_onnx_upper = conv_mod.convert_to_onnx(train_x, model_result['model_upper'], modelreport_json)
                model_onnx_lower = conv_mod.convert_to_onnx(train_x, model_result['model_lower'], modelreport_json)
                
                model_onnx_upper_name = onnx_model_name[:-5]+'_upper_limit.onnx'
                model_onnx_lower_name = onnx_model_name[:-5]+'_lower_limit.onnx'

                with open("mlflow_tmp/"+model_onnx_upper_name, "wb") as f:
                        f.write(model_onnx_upper.SerializeToString()) 

                mlflow.log_artifact("mlflow_tmp/"+model_onnx_upper_name, artifact_path="model")

                with open("mlflow_tmp/"+model_onnx_lower_name, "wb") as f:
                        f.write(model_onnx_lower.SerializeToString()) 

                mlflow.log_artifact("mlflow_tmp/"+model_onnx_lower_name, artifact_path="model")



            if model_type in ["LinearRegression","DecisionTreeRegressor","ElasticNet","PolyRegression"]:
                print('convert onnx')
                model_onnx = conv_mod.convert_to_onnx(train_x, model_result['model'], modelreport_json)


            with open("mlflow_tmp/"+onnx_model_name, "wb") as f:
                    f.write(model_onnx.SerializeToString()) 

            mlflow.log_artifact("mlflow_tmp/"+onnx_model_name, artifact_path="model")
                
            mlflow.log_dict(json.loads(modelreport_json),"Rapports/rapport_modelisation.json")



            date_debut = datetime.datetime.strftime(data.index[0], "%d-%b-%Y")
            date_fin   = datetime.datetime.strftime(data.index[-1], "%d-%b-%Y")
            periode    = date_debut + ' Au ' + date_fin

            mlflow.set_tag("Période",periode)
            mlflow.set_tag("Type",model_result['model_type'])
            mlflow.set_tag("Nombre échantillons",data.shape[0])
            mlflow.set_tag("Fréquence",dico_model['freq'])

            list_fact = [f['nom'] for t,f in dico_model["facteurs"].items() if f['used']]
            string_fact = ' + '.join(list_fact)
            mlflow.set_tag("Facteurs",string_fact)
    
            # Suppresion des fichiers temporaires

            for f in os.listdir("mlflow_tmp/"):
                    os.remove(os.path.join("mlflow_tmp/", f))



    mlflow.end_run()
    return 


def Log_Params(dico_params, mlflow):
    if dico_params is not None:
        for p_name, p in dico_params.items():
            if p is not None:
                mlflow.log_param(p_name,p)

def Log_Figures(dico_figure, mlflow):
    for f_name, fig in dico_figure.items():
        if fig is not None:
            mlflow.log_figure(fig,'images/'+f_name + '.png')


def Log_Metrics(dico_metrics, mlflow):
    for m_name, m in dico_metrics.items():
        if m is not None:
            mlflow.log_metric(m_name,m)


def Log_Data(model_result, mlflow):

    if model_result['test_data_set'] is not None:

        model_result['test_data_set'].to_csv('mlflow_tmp/test_data_set.csv')
        mlflow.log_artifact('mlflow_tmp/test_data_set.csv', artifact_path="data")
        model_result['train_data_set'].to_csv('mlflow_tmp/train_data_set.csv')
        mlflow.log_artifact('mlflow_tmp/train_data_set.csv', artifact_path="data") 

    else:

        model_result['train_data_set'].to_csv('mlflow_tmp/train_data_set.csv')
        mlflow.log_artifact('mlflow_tmp/train_data_set.csv', artifact_path="data")


def Log_Facteur(dico_model, mlflow):

    from pretty_html_table import build_table

    data_fact = {'Facteurs':[f['nom'] for tag, f in dico_model['facteurs'].items()],
                 'Description':[f['description'] for tag, f in dico_model['facteurs'].items()],
                 'Unité':[f['unit'] for tag, f in dico_model['facteurs'].items()],
                 'Utilisé':['oui' if f['used'] else 'non' for tag, f in dico_model['facteurs'].items()]}
    df_data_fact = pd.DataFrame(data=data_fact)

    html_table_blue_light = build_table(df_data_fact, 'blue_light')

    # Save to html file
    with open('mlflow_tmp/facteurs.html', 'w', encoding="utf-8") as f:
        f.write(html_table_blue_light)

    mlflow.log_artifact("mlflow_tmp/facteurs.html","Facteurs")




def Log_Stat_Desc(dico_model, mlflow, data):

    from pretty_html_table import build_table


    dico_map_name = {d['nom']:d['description'] for d in dico_model["facteurs"].values() if d['used']}
    dico_map_name[dico_model["tag_name"]] = dico_model["tag_name"]
    data_desc = data.describe()

    dico_map_unit = {d['nom']:d['unit'] for d in dico_model["facteurs"].values() if d['used']}
    dico_map_unit[dico_model["tag_name"]] = dico_model["tag_unit"]
    data_desc = data_desc.T
    data_desc['unit'] = ''
    data_desc['description'] = ''
    for name,row in data_desc.iterrows():
        data_desc.loc[name,'description'] = dico_map_name[name]
        data_desc.loc[name,'unit'] = dico_map_unit[name]
    data_desc = data_desc[['description','count','unit','mean','min','max']]
    data_desc.columns = ['Variable','nombre','unite','moyenne','min','max']
    data_desc['nombre'] = data_desc['nombre'].astype(int)

    html_table_blue_light = build_table(data_desc, 'blue_light')

    # Save to html file
    with open('mlflow_tmp/statdesc.html', 'w', encoding="utf-8") as f:
        f.write(html_table_blue_light)

    mlflow.log_artifact("mlflow_tmp/statdesc.html","Statistiques")



def Log_Model_Summary(model_result, mlflow):

    from pretty_html_table import build_table


    idx_table = ['Type modele'] 
    value_table = [model_result['model_type']]

    for metric_name, metric_value in model_result['metrics'].items():
        if metric_value is not None:
            idx_table.append(metric_name)
            if 'mape' in metric_name:
                value_table.append(str(round(metric_value,2))+ ' %')
            else:
                value_table.append(str(round(metric_value,2)))

    idx_table.append('Formule')
    value_table.append(model_result['formula'])

    pd_df = pd.DataFrame(index=idx_table,data={'champ':idx_table,'valeur':value_table})

    html_table_blue_light = build_table(pd_df, 'blue_light')

    # Save to html file
    with open('mlflow_tmp/model_summary.html', 'w', encoding="utf-8") as f:
        f.write(html_table_blue_light)

    mlflow.log_artifact("mlflow_tmp/model_summary.html","Résumé du Modèle")

    if not model_result['pertinence'].empty:

        html_table_pert = build_table(model_result['pertinence'], 'blue_light')

        # Save to html file
        with open('mlflow_tmp/pertinence_facts.html', 'w', encoding="utf-8") as f:
            f.write(html_table_pert)

        mlflow.log_artifact("mlflow_tmp/pertinence_facts.html","Résumé du Modèle")    






def Load_Information_For_Report(run):

    import commentjson

    rep_artifact = run.info.artifact_uri.replace('file:///X','X')

    dico_infos_location = dict()

    import os
    import commentjson


    for it in os.scandir(rep_artifact):
        if it.is_dir():
            if 'images' in it.path:
                dico_infos_location['images'] = dict()

                for fimage in os.scandir(it.path):
                    dico_infos_location['images'][fimage.name] = fimage.path

            if 'interprétation' in it.path:
                dico_infos_location['interp'] = dict()

                for fint in os.scandir(it.path):
                    if '.png' in fint.name:
                        dico_infos_location['interp'][fint.name] = fint.path

            if 'Rapports' in it.path:
                dico_infos_location['rapport'] = dict()
    
                for frapport in os.scandir(it.path):
                    if '.json' in frapport.name:
                        #dico_infos_location['rapport'][frapport.name] = frapport.path
                            
                        with open(frapport.path, encoding='utf-8') as file:
                            dico_infos_location['rapport'] = commentjson.load(file)


            if 'Rapports' in it.path:
                dico_infos_location['rapport'] = dict()
    
                for frapport in os.scandir(it.path):
                    if '.json' in frapport.name:
                        #dico_infos_location['rapport'][frapport.name] = frapport.path
                            
                        with open(frapport.path, encoding='utf-8') as file:
                            dico_infos_location['rapport'] = commentjson.load(file)

        if it.is_file():
            if it.name == 'model_config.json':            
                with open(it.path, encoding='utf-8') as file:
                    dico_infos_location['config'] = commentjson.load(file)
            
            

    return dico_infos_location



def Load_Information_From_run(run):

    import commentjson
    import pandas as pd
    import mlflow
    from datetime import datetime

    dico_model_data = dict()

    date_parser = lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S")

    train_data_set_file = mlflow.artifacts.download_artifacts(run_id=run.info.run_id,artifact_path="data/train_data_set.csv")
    train_data_set = pd.read_csv(train_data_set_file)
    train_data_set["Date"] = train_data_set["Date"].apply(date_parser)
    train_data_set.set_index('Date',inplace=True)

    try:

        test_data_set_file  = mlflow.artifacts.download_artifacts(run_id=run.info.run_id,artifact_path="data/test_data_set.csv")

    except OSError as error:
        test_data_set_file = None

    if test_data_set_file is not None:
        test_data_set  = pd.read_csv(test_data_set_file)
        test_data_set["Date"] = test_data_set["Date"].apply(date_parser)
        test_data_set.set_index('Date',inplace=True)

        data = pd.concat([train_data_set,test_data_set])
        data.sort_index(inplace=True)

    else:
        data = train_data_set

    
    dir_mod = "runs:/"+run.info.run_id+"/model"
    model = mlflow.sklearn.load_model(dir_mod)

    model_config_file = mlflow.artifacts.download_artifacts(run_id=run.info.run_id,artifact_path="model_config.json")
    with open(model_config_file, encoding='utf-8') as file:
        model_config= commentjson.load(file)

    dico_model_data['data'] = data
    dico_model_data['model'] = model
    dico_model_data['model_config'] = model_config

    return dico_model_data

def get_hp_parameters_from_run(run):

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
    return hp_params   

def Draw_Metrics_From_Exp(experiment):

    import pandas as pd
    import mlflow
    from mlflow.tracking import MlflowClient
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import plotly.express as px

    experiment_mlf = mlflow.get_experiment_by_name(experiment)

    runs = MlflowClient().search_runs(
        experiment_ids = experiment_mlf.experiment_id)

    df_metrics = pd.DataFrame(columns=['Run','R2','Erreur Relative','Phase'])

    for run in runs:

        mlflow_R2_train = MlflowClient().get_metric_history(run.info.run_id,'R2_train')
        mlflow_R2_test = MlflowClient().get_metric_history(run.info.run_id,'R2_test')
        mlflow_mape_train = MlflowClient().get_metric_history(run.info.run_id,'mape_train')
        mlflow_mape_test = MlflowClient().get_metric_history(run.info.run_id,'mape_test')
        
        new_serie = pd.Series([run.info.run_name,100*mlflow_R2_train[0].value,mlflow_mape_train[0].value,'Apprentissage'],index=df_metrics.columns)
        df_metrics = df_metrics.append(new_serie,ignore_index=True)

        if len(mlflow_R2_test) > 0:
  
            #df_metrics = pd.concat([df_metrics,pd.DataFrame(new_row)])
            new_serie = pd.Series([run.info.run_name,100*mlflow_R2_test[0].value,mlflow_mape_test[0].value,'Validation'],index=df_metrics.columns)
            df_metrics = df_metrics.append(new_serie,ignore_index=True)

    fig = px.histogram(df_metrics, x="Run", y="R2",
                color='Phase', barmode='group',
                color_discrete_map={
                        'Apprentissage': 'blue',
                        'Validation': 'green'
                    },
                height=400)

    fig.update_layout(
        xaxis_title="modèle", yaxis_title="R2")
    fig.show()

    fig2 = px.histogram(df_metrics, x="Run", y="Erreur Relative",
                color='Phase', barmode='group',
                color_discrete_map={
                        'Apprentissage': 'blue',
                        'Validation': 'green'
                    },
                height=400)

    fig2.update_layout(
        xaxis_title="modèle", yaxis_title="Erreur Relative")
    fig2.show()
    
'''

def Load_Information_From_run(run):

    import pandas as pd
    import mlflow
    import commentjson

    dico_model_data = dict()





    train_data_set_file = mlflow.artifacts.download_artifacts(run_id=run.info.run_id,artifact_path="data/train_data_set.csv")
    test_data_set_file  = mlflow.artifacts.download_artifacts(run_id=run.info.run_id,artifact_path="data/test_data_set.csv")


    date_parser = lambda x: pd.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S")

    train_data_set = pd.read_csv(train_data_set_file,parse_dates=["Date"],date_parser=date_parser,index_col=0)
    test_data_set  = pd.read_csv(test_data_set_file,parse_dates=["Date"],date_parser=date_parser,index_col=0)
    data = pd.concat([train_data_set,test_data_set])
    data.sort_index(inplace=True)
    
    dir_mod = "runs:/"+run.info.run_id+"/model"
    model = mlflow.sklearn.load_model(dir_mod)

    model_config_file = mlflow.artifacts.download_artifacts(run_id=run.info.run_id,artifact_path="model_config.json")
    with open(model_config_file, encoding='utf-8') as file:
        model_config= commentjson.load(file)

    dico_model_data['data'] = data
    dico_model_data['model'] = model
    dico_model_data['model_config'] = model_config

    return dico_model_data

'''