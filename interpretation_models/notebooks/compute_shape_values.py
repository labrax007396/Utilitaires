import pandas as pd
import os,sys,importlib
import sklearn, lightgbm
path = os.getcwd()
path_src = os.path.abspath(os.path.join(path, os.pardir,"src"))
sys.path.append(path_src)


from importdata import import_from_influxdb
importlib.reload(import_from_influxdb)
from modelisation import interpreter_models

import argparse


def read_model_config(file_config:str)->dict:
    import commentjson

    MANDATORY_KEYS = ["tag_modelise","tag_name","facteurs"]
    MANDATORY_FACTEUR_KEYS = ["used","type","nom"]

    message_error = list()


    try:
        with open(file_config, encoding='utf-8') as file:
            dico_model = commentjson.load(file)
    except Exception as error:
        message_error.append(type(error).__name__ +" :Le fichier model_config.json est incorrectement formaté")
        return  None, message_error

    for mk in MANDATORY_KEYS:
        if mk not in dico_model.keys():
            message_error.append("Le champs " + mk + " est absent du fichier de configuration du modèle")
            dico_model =  None
            return  None, message_error

    try:         
        int(dico_model['tag_modelise'][4:])
    except ValueError as ve:
        message_error.append("La valeur du champ tag_modelise (" + dico_model['tag_modelise'] + ") " + "doit être au format 'tag_unentier'")
        return  None, message_error        

    # Vérification des clés des facteurs:
    for tf in dico_model['facteurs'].keys():
        try:         
            int(tf[4:])
        except ValueError as ve:
            message_error.append("La valeur du champ des facteurs  (" + tf + ") " + "doit être au format 'tag_unentier'")
            return  None, message_error         
    
    for t,f in dico_model['facteurs'].items():
        for mf in MANDATORY_FACTEUR_KEYS:
            if mf not in f.keys():
                message_error.append("Le champ: " + mf + " est absent pour le facteur " + t)
                return  None, message_error 

    return  dico_model, message_error



def read_data(dico_model:dict,start_date:str,end_date:str) -> pd.DataFrame:
    import datetime

    start_date = datetime.datetime.strptime(start_date, '%d/%m/%Y %H:%M:%S').isoformat()
    end_date   = datetime.datetime.strptime(end_date, '%d/%m/%Y %H:%M:%S').isoformat()


    data, clean_report ,read_message_error = import_from_influxdb.Charger_Preparer_Data(ref_periode_debut = start_date, 
                                            ref_periode_fin   = end_date,
                                            ipe_tag           = dico_model['tag_modelise'],
                                            dico_du_model     = dico_model,
                                            use_seuil_min     = True,
                                            use_seuil_max     = True,
                                            clean_data        = False,                            
                                            concat_after      = True,
                                            load_unused_feature = False,
                                            zscore            = 3)
    

    return data, read_message_error

def read_pkl(model_file_pkl:str):
    import pickle
    model = pickle.load(open(model_file_pkl, 'rb'))
    return model

def check_facteurs_name(model,data):

    """ 
        Vérification que les noms des facteurs des données coïncident
        avec ceux utilisés dans le modèle

    """
    message_error = list()

    features_data = set(data.columns[1:])

    if type(model) == sklearn.pipeline.Pipeline: # Le modèle est un Pipeline 
        features_model = set(model.steps[0][1].feature_names_in_)
    else:
        features_model = set(model.feature_names_in_)

    features_data_not_in_model = features_data - features_model
    features_model_not_in_data = features_model - features_data

    if len(features_data_not_in_model) > 0 :
        message = 'Le(s) facteur(s) ' + ' '.join(list(features_data_not_in_model))  + ' Ne sont pas dans le modèle'
        message_error.append(message)
    if len(features_model_not_in_data) > 0 :
        message = 'Le(s) facteur(s) ' + ' '.join(list(features_model_not_in_data))  + ' Ne sont pas dans les données'
        message_error.append(message)

    return message_error


def check_argument(args):

    import datetime

    DATE_FORMAT = '%d/%m/%Y %H:%M:%S'

    message_error = list()

    try:
        start_date = datetime.datetime.strptime(args.start_date, DATE_FORMAT).isoformat()
    except ValueError as ve:

        message = "La date de début: " + args.start_date + ''' n'est pas au format ''' + DATE_FORMAT
        message_error.append(message)
        return message_error

    try:
        end_date = datetime.datetime.strptime(args.end_date, DATE_FORMAT).isoformat()
    except ValueError as ve:

        message = "La date de fin: " + args.end_date + ''' n'est pas au format ''' + DATE_FORMAT
        message_error.append(message)
        return message_error

    if end_date <= start_date:
        message = "La date de début: " + args.start_date + " doit être antérieure à la date de fin: " + args.end_date
        message_error.append(message)
        return message_error
    
    else:
        return list()
    
def log_message(message):
    f = open("shape_calculation.log", "a")
    for m in message:
        f.write(m)        
    f.close()    

def main():

    if os.path.exists("shape_calculation.log"):
        os.remove("shape_calculation.log")

    if os.path.exists("shape_values.csv"):
        os.remove("shape_values.csv")

    parser = argparse.ArgumentParser(description='Arguments calcul des shapes values')
    parser.add_argument('model_file_config', type=str,help='emplacement du fichier de configuration du modèle')
    parser.add_argument('start_date', type=str,help='date début au format d/M/Y H:M:S')
    parser.add_argument('end_date', type=str,help='date fin au format d/M/Y H:M:S')
    parser.add_argument('model_file_pkl', type=str,help='emplacement du fichier modèle au format pkl')

    args = parser.parse_args()
    message_error_input_args = check_argument(args)

    if len(message_error_input_args):
        log_message(message_error_input_args)
        return
   
    dico_model, message_error = read_model_config(file_config = args.model_file_config)

    if len(message_error):
        log_message(message_error)
        return

   
    data, read_message_error = read_data(dico_model = dico_model,start_date = args.start_date,end_date = args.end_date)
    if len(read_message_error):
        log_message(read_message_error)
        return


    model = read_pkl(model_file_pkl = args.model_file_pkl)

    facteurs_missing_message_error  = check_facteurs_name(model,data)
    
    if len(facteurs_missing_message_error):
        log_message(facteurs_missing_message_error)
        return
    
    dico_model_data = {'data':data,
                       'model':model,
                       'model_config':dico_model}

    svals, df_x = interpreter_models.Compute_Shape_Values(dico_model_data = dico_model_data)

    df_svals = pd.DataFrame(index = df_x.index,data=svals,columns=df_x.columns)

    df_svals.to_csv("shape_values.csv")
    
if __name__ == "__main__":
    
    main()
#python compute_shape_values.py "data_test/model_config.json" "09/04/2023  00:00:00" "14/04/2023  11:00:00" "data_test/model.pkl"