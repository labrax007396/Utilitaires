    ### Paramétrisation de l'experience ###########################################

import datetime
import commentjson

def Experiment_Params(exp_config,mlflow) -> dict:

    dico_exp = dict()

    experiment  = exp_config['client'] + "-" + exp_config['site'] + "-" + exp_config['depart'] + "-"  + exp_config['case_study']

    dico_exp['experiment'] = experiment

    ###############################################################


    dico_exp['dir_case']  = exp_config['dir_models'] + exp_config['client'] + "/" + exp_config['site'] + "/" + exp_config['depart'] + "/CONFIGS/" + exp_config['case_study'] + "/"


    dico_exp['file_model_param']  = dico_exp['dir_case'] + "model_config.json"

    ###############################################################

    ##dico_exp['rep_tracking_mlflow'] = "file:///"+exp_config['dir_models']+"MLFLOW"
    dico_exp['rep_tracking_mlflow'] = "file:///"+exp_config['dir_models']+exp_config['client']+'/'+exp_config['site']+ "/" + exp_config['depart'] +'/'+'ARCTIFACTS'


    mlflow.set_tracking_uri(dico_exp['rep_tracking_mlflow'])

    dico_exp['dico_figure'] = dict() # dictionnaire des figures qui seront logguées 

    dico_exp['model_pkl']   = dico_exp['dir_case'] + "model.pkl"

    with open(dico_exp['file_model_param'], encoding='utf-8') as file:
        dico_model = commentjson.load(file)

    feat_desc_dup, feat_name_dup = Check_Name_Description(dico_model=dico_model)

    if len(feat_desc_dup)>0:
        print("Les descriptions des facteurs suivants sont dupliquées: ",feat_desc_dup)
        return None
    
    if len(feat_name_dup)>0:
        print("Les noms des facteurs suivants sont dupliqués: ",feat_name_dup)
        return None
    

    dico_exp['uv_mangling']         = dico_model['mangling']
    dico_exp['site']                = dico_model['site']
    dico_exp['onnx_model_name'] = exp_config['case_study'] + '.onnx'

    dico_exp['dico_model'] = dico_model

    dico_exp['ref_periode_debut'] = datetime.datetime.strptime(dico_model['ref_periode_debut'], '%d/%m/%Y %H:%M:%S').isoformat()
    dico_exp['ref_periode_fin']   = datetime.datetime.strptime(dico_model['ref_periode_fin'], '%d/%m/%Y %H:%M:%S').isoformat()

    return dico_exp

def Reload_Model_Config(dico_exp:dict) -> dict:

    with open(dico_exp['file_model_param'], encoding='utf-8') as file:
        dico_model = commentjson.load(file)
    
    dico_exp['dico_model'] = dico_model

    return dico_exp

def Check_Name_Description(dico_model:dict):

    facteurs = dico_model['facteurs']
    list_feat_name = [dfact['nom'] for dfact in facteurs.values()]
    list_feat_desc = [dfact['description'] for dfact in facteurs.values()]

    feat_desc_dup = [x for i, x in enumerate(list_feat_desc) if i != list_feat_desc.index(x)]
    feat_name_dup = [x for i, x in enumerate(list_feat_name) if i != list_feat_name.index(x)]

    return feat_desc_dup, feat_name_dup