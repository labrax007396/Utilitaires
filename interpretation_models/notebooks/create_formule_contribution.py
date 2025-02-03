import pandas as pd
import csv
import os,sys,importlib
path = os.getcwd()
path_src = os.path.abspath(os.path.join(path, os.pardir,"src"))
sys.path.append(path_src)


from importdata import import_from_influxdb
importlib.reload(import_from_influxdb)

import argparse

ScopeMangling = list()
Description   = list()
Identifiant   = list()
Frequence     = list()
Roles         = list()
Type_donnees  = list()
Unites        = list()
Formule       = list()


def create_tags_models(contrib_config:dict, formula_file_name:str, save_formula_file:bool) -> pd.DataFrame:

    """ 
        Création des formules pour les tags des contributions des facteurs
        Pour l'instant la formule = 0 car on ne sais pas créer des formules pour des modèles onnx
    
    """


    for n_contrib,d_contrib in contrib_config['contribution'].items():
        if type(d_contrib) == list:

            formula_groupe = ''
            for sub_name in d_contrib:
                ScopeMangling.append(contrib_config['dir_interp'])
                Description.append(sub_name)
                Identifiant.append(sub_name)
                Frequence.append(contrib_config['frequence'])
                Roles.append('NormalizationRecalc')
                Type_donnees.append('Numeric')
                Unites.append(contrib_config['Unit'])
                formula = '0'
                Formule.append(formula)

                formula_groupe = formula_groupe + '['+ contrib_config['dir_interp'] + '.' + sub_name + '.' + contrib_config['frequence'] + '.NormalizationRecalc]+'

            ScopeMangling.append(contrib_config['dir_interp'])
            Description.append(n_contrib)
            Identifiant.append(n_contrib)
            Frequence.append(contrib_config['frequence'])
            Roles.append('NormalizationRecalc')
            Type_donnees.append('Numeric')
            Unites.append(contrib_config['Unit'])
            formula = formula_groupe[0:-1]
            Formule.append(formula)        

        else:
            ScopeMangling.append(contrib_config['dir_interp'])
            Description.append(d_contrib)
            Identifiant.append(n_contrib)
            Frequence.append(contrib_config['frequence'])
            Roles.append('NormalizationRecalc')
            Type_donnees.append('Numeric')
            Unites.append(contrib_config['Unit'])
            formula = '0'
            Formule.append(formula)        

    if save_formula_file:
        dico_formula = {'ScopeMangling':ScopeMangling,
                        'Description':Description,
                        'Fréquence':Frequence,     
                        'Rôles':Roles,                    
                        'Identifiant':Identifiant,
                        'Type de données':Type_donnees,
                        'Unités':Unites,
                        'Formule':Formule}

        df_formules = pd.DataFrame(data=dico_formula)
        df_formules.to_csv(contrib_config['dir_formula']+formula_file_name,index=False, sep=";", quoting=csv.QUOTE_NONE)

    return df_formules



def create_tags_erreur_estimation(contrib_config:dict, 
                                  formula_file_name:str, 
                                  save_formula_file:bool) -> pd.DataFrame:

    """ 
        Création des formules pour:
        - La valeur de référence de l'IPe
        - L'erreur entre le modèle onnx de l'IPe et le modèle obtenu avec les shapes values
    
    """

    import datetime

    # valeur moyenne de référence

    tag_mesure = contrib_config['tag_mesure']


    ref_periode_debut  = datetime.datetime.strptime(contrib_config['debut_ref'], '%Y-%m-%d %H:%M:%S').isoformat()
    ref_periode_fin    = datetime.datetime.strptime(contrib_config['fin_ref'], '%Y-%m-%d %H:%M:%S').isoformat()

    mesure = import_from_influxdb.GetDataFromUV(tag_id = [tag_mesure], 
                                            ref_periode_debut = ref_periode_debut, 
                                            ref_periode_fin   = ref_periode_fin)
    
    txt_ref = "Valeur de référence {ref_value:.2f} " + contrib_config['Unit']
    print(txt_ref.format(ref_value = mesure[tag_mesure].mean()))


    ScopeMangling = list()
    Description   = list()
    Identifiant   = list()
    Frequence     = list()
    Roles         = list()
    Type_donnees  = list()
    Unites        = list()
    Formule       = list()

    # Formule pour la valeur de référence

    ScopeMangling.append(contrib_config['parentmgl_ipe'])
    Description.append(contrib_config['des_ipe'])
    Identifiant.append(contrib_config['nom_ipe'])
    Frequence.append(contrib_config['frequence'])
    Roles.append('Validation')
    Type_donnees.append('Numeric')
    Unites.append(contrib_config['Unit'])
    signature_mesure = contrib_config['parentmgl_ipe']+'.'+contrib_config['nom_ipe']+'.'+contrib_config['frequence']
    formula = str(mesure[tag_mesure].mean()) + '*(1.0 + 0*[' + signature_mesure + '])'
    Formule.append(formula)   

    signature_reference = signature_mesure + '.Validation'
    signature_modele    = signature_mesure + '.NormalizationRecalc'

    # Formule pour l'erreur d'estimation

    formule_erreur = '['+signature_reference+']-['+signature_modele+']+'


    for n_contrib,d_contrib in contrib_config['contribution'].items():
        if type(d_contrib) == list:
            desc = n_contrib
            iden = n_contrib
        else:
            desc = d_contrib
            iden = n_contrib

        signature_contrib = contrib_config['dir_interp'] + '.' + iden + '.' + contrib_config['frequence'] + '.NormalizationRecalc'

        formule_erreur += '[' + signature_contrib + ']+'

    formule_erreur = formule_erreur[:-1]

    ScopeMangling.append(contrib_config['dir_interp'])
    Description.append("Erreur estimation")
    Identifiant.append("Erreur_estimation")
    Frequence.append(contrib_config['frequence'])
    Roles.append('Data')
    Type_donnees.append('Numeric')
    Unites.append(contrib_config['Unit'])
    Formule.append(formule_erreur) 


    if save_formula_file:
        dico_formula = {'ScopeMangling':ScopeMangling,
                        'Description':Description,
                        'Fréquence':Frequence,     
                        'Rôles':Roles,                    
                        'Identifiant':Identifiant,
                        'Type de données':Type_donnees,
                        'Unités':Unites,
                        'Formule':Formule}

        df_formules = pd.DataFrame(data=dico_formula)
        df_formules.to_csv(contrib_config['dir_formula']+formula_file_name,index=False, sep=";", quoting=csv.QUOTE_NONE)

    return df_formules



def create_formules_contribu_ajustees(contrib_config:dict, 
                                      formula_file_name:str, 
                                      save_formula_file:bool) -> pd.DataFrame:

    """ 
        Création des formules pour:
        - Les contributions ajustées des facteurs
    
    """

    ScopeMangling = list()
    Description   = list()
    Identifiant   = list()
    Frequence     = list()
    Roles         = list()
    Type_donnees  = list()
    Unites        = list()
    Formule       = list()

    ScopeMangling.append(contrib_config['dir_interp'])
    Description.append("Somme contributions")
    Identifiant.append("Somme_contributions")
    Frequence.append(contrib_config['frequence'])
    Roles.append('Data')
    Type_donnees.append('Numeric')
    Unites.append(contrib_config['Unit'])

    formule = ''
    for n_contrib in contrib_config['contribution'].keys():
        signature_contrib = contrib_config['dir_interp'] + '.' + n_contrib + '.' + contrib_config['frequence'] + '.NormalizationRecalc'
        formule += '[' + signature_contrib + '].Abs()+'
    formule = formule[:-1]
    Formule.append(formule) 



    signature_somme_contrib = contrib_config['dir_interp'] + '.Somme_contributions' + '.' + contrib_config['frequence']
    signature_erreur        = contrib_config['dir_interp']+'.Erreur_estimation.'+contrib_config['frequence']


    for n_contrib,d_contrib in contrib_config['contribution'].items():
        
        if type(d_contrib) == list:
            desc = n_contrib
            iden = n_contrib
        else:
            desc = d_contrib
            iden = n_contrib

        ScopeMangling.append(contrib_config['dir_interp'])
        Description.append(desc)
        Identifiant.append(iden)
        Frequence.append(contrib_config['frequence'])
        Roles.append('Data')
        Type_donnees.append('Numeric')
        Unites.append(contrib_config['Unit'])
        
        signature_contrib = contrib_config['dir_interp'] + '.' + n_contrib + '.' + contrib_config['frequence'] + '.NormalizationRecalc'
        formule = '['+signature_contrib +']-['+signature_erreur +']*' + '['+signature_contrib+'].Abs()/[' + signature_somme_contrib + ']'
        Formule.append(formule) 

    if save_formula_file:
        dico_formula = {'ScopeMangling':ScopeMangling,
                        'Description':Description,
                        'Fréquence':Frequence,     
                        'Rôles':Roles,                    
                        'Identifiant':Identifiant,
                        'Type de données':Type_donnees,
                        'Unités':Unites,
                        'Formule':Formule}

        df_formules = pd.DataFrame(data=dico_formula)
        df_formules.to_csv(contrib_config['dir_formula']+formula_file_name,index=False, sep=";", quoting=csv.QUOTE_NONE)

    return df_formules



def create_formules_contribu_frequences(contrib_config:dict, 
                                      formula_file_name:str, 
                                      save_formula_file:bool) -> pd.DataFrame:

    """ 
        Création des formules pour:
        - Les contributions ajustées des facteurs pour les fréquences souhaitées
    
    """

    ScopeMangling = list()
    Description   = list()
    Identifiant   = list()
    Frequence     = list()
    Roles         = list()
    Type_donnees  = list()
    Unites        = list()
    Formule       = list()

    signature_mesure        = contrib_config['parentmgl_ipe']+'.'+contrib_config['nom_ipe']+'.'+contrib_config['frequence']
    signature_reference     = signature_mesure + '.Validation'

    for f in contrib_config['list_freq']:

        if f == "ShiftWork":
            ft = "8h"
        elif f == "Day":
            ft = "1d"
        elif f == "Week":
            ft = "1w"
        elif f == "Month":
            ft = "1M"
        elif f == "Year":
            ft = "1y"

        offset = contrib_config['offset']

        # Formule pour la valeur de référence

        ScopeMangling.append(contrib_config['dir_interp'])
        Description.append(contrib_config['des_ref'])
        Identifiant.append(contrib_config['nom_ref'])
        Frequence.append(f)
        Roles.append('Validation')
        Type_donnees.append('Numeric')
        Unites.append(contrib_config['Unit'])
        
        formula = '['+signature_reference+'].Average("' + ft + '","' + offset + '")'
        Formule.append(formula)

        # Formule pour la mesure

        ScopeMangling.append(contrib_config['dir_interp'])
        Description.append(contrib_config['des_ref'])
        Identifiant.append(contrib_config['nom_ref'])
        Frequence.append(f)
        Roles.append('Data')
        Type_donnees.append('Numeric')
        Unites.append(contrib_config['Unit'])
        
        formula = '['+signature_mesure+'].Average("' + ft + '","' + offset + '")'
        Formule.append(formula)    

        # Formule pour les contributions et Autre

        signature_mesure_f        = contrib_config['dir_interp']+'.'+contrib_config['nom_ref']+'.'+f
        signature_reference_f     = contrib_config['dir_interp']+'.'+contrib_config['nom_ref']+'.'+f + '.Validation'

        formula_autre = '['+signature_mesure_f + ']-[' + signature_reference_f+']-('

    
        for n_contrib,d_contrib in contrib_config['contribution'].items():
            
            if type(d_contrib) == list:
                desc = n_contrib
            else:
                desc = d_contrib

            signature_contrib = contrib_config['dir_interp'] + '.' + n_contrib + '.' + contrib_config['frequence'] 

            ScopeMangling.append(contrib_config['dir_interp'])
            Description.append(desc)
            Identifiant.append(n_contrib)
            Frequence.append(f)
            Roles.append('Data')
            Type_donnees.append('Numeric')
            Unites.append(contrib_config['Unit'])

            formula = '['+signature_contrib+'].Average("' + ft + '","' + offset + '")'
            Formule.append(formula)

            # formule pour Autres

            signature_contrib_f = contrib_config['dir_interp'] + '.' + n_contrib + '.' + f

            formula_autre += '[' + signature_contrib_f + ']+'
        
        formula_autre = formula_autre[:-1] + ')'
        ScopeMangling.append(contrib_config['dir_interp'])
        Description.append('Autre')
        Identifiant.append('Autre')
        Frequence.append(f)
        Roles.append('Data')
        Type_donnees.append('Numeric')
        Unites.append(contrib_config['Unit'])
        Formule.append(formula_autre)    

    if save_formula_file:
        dico_formula = {'ScopeMangling':ScopeMangling,
                        'Description':Description,
                        'Fréquence':Frequence,     
                        'Rôles':Roles,                    
                        'Identifiant':Identifiant,
                        'Type de données':Type_donnees,
                        'Unités':Unites,
                        'Formule':Formule}

        df_formules = pd.DataFrame(data=dico_formula)
        df_formules.to_csv(contrib_config['dir_formula']+formula_file_name,index=False, sep=";", quoting=csv.QUOTE_NONE)

    return df_formules


def create_formules_contribu_previous_period(contrib_config:dict, 
                                      formula_file_name:str, 
                                      save_formula_file:bool) -> pd.DataFrame:
    


    ScopeMangling = list()
    Description   = list()
    Identifiant   = list()
    Frequence     = list()
    Roles         = list()
    Type_donnees  = list()
    Unites        = list()
    Formule       = list()


    for f in contrib_config['list_freq']:

        if f == "ShiftWork":
            ft = "8h"
        elif f == "Day":
            ft = "1d"
        elif f == "Week":
            ft = "1w"
        elif f == "Month":
            ft = "1M"
        elif f == "Year":
            ft = "1y"

        # Formule pour la valeur de l'IPé à la période précédente

        signature_mesure = contrib_config['dir_interp']+'.'+contrib_config['nom_ref']+'.'+f

        ScopeMangling.append(contrib_config['dir_interp'])
        Description.append(contrib_config['des_ref']+ ' Période précédente')
        Identifiant.append(contrib_config['nom_ref']+'_Previous')
        Frequence.append(f)
        Roles.append('Data')
        Type_donnees.append('Numeric')
        Unites.append(contrib_config['Unit'])

        formula = '['+signature_mesure+'].AddPeriod("' + ft + '")'
        Formule.append(formula)

        # Formule pour les variations des contributions

        for n_contrib,d_contrib in contrib_config['contribution'].items():
            
            if type(d_contrib) == list:
                desc = n_contrib
            else:
                desc = d_contrib

            signature_contrib = contrib_config['dir_interp'] + '.' + n_contrib + '.' + f 

            ScopeMangling.append(contrib_config['dir_interp'])
            Description.append(desc)
            Identifiant.append(n_contrib)
            Frequence.append(f)
            Roles.append('HighestDifference')
            Type_donnees.append('Numeric')
            Unites.append(contrib_config['Unit'])

            formula = '['+signature_contrib+']-['+signature_contrib+'].AddPeriod("' + ft + '")' 
            Formule.append(formula)

        # formule pour variation de "Autres"

        signature_autre = contrib_config['dir_interp'] + '.Autre.' + f

        formula_autre  = '['+signature_autre+']-['+signature_autre+'].AddPeriod("' + ft + '")' 

        ScopeMangling.append(contrib_config['dir_interp'])
        Description.append('Autre')
        Identifiant.append('Autre')
        Frequence.append(f)
        Roles.append('HighestDifference')
        Type_donnees.append('Numeric')
        Unites.append(contrib_config['Unit'])
        Formule.append(formula_autre)    

    if save_formula_file:
        dico_formula = {'ScopeMangling':ScopeMangling,
                        'Description':Description,
                        'Fréquence':Frequence,     
                        'Rôles':Roles,                    
                        'Identifiant':Identifiant,
                        'Type de données':Type_donnees,
                        'Unités':Unites,
                        'Formule':Formule}

        df_formules = pd.DataFrame(data=dico_formula)
        df_formules.to_csv(contrib_config['dir_formula']+formula_file_name,index=False, sep=";", quoting=csv.QUOTE_NONE)

    return df_formules