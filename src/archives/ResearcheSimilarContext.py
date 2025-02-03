import pandas as pd 


def RechercheContextSimilaires(data = None,
                               dico_actual_situation = None,
                               weight_factors = None,
                               nbre_situations = 10):

    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    actual_date =  dico_actual_situation['actual_date']

    data_recherche = data.copy()

    ### on supprime la situation actuelle dans le champs d'investigation
    
    data_recherche.drop(pd.Timestamp(actual_date),inplace=True)

    ### Si une variable cat est présente dans les Facteurs, on restreind le champ des recherches

    num_feat = list() # Liste de facteurs numériques

    for feat, value in dico_actual_situation['facteurs'].items():
        if type(value) == str:
            data_recherche = data_recherche[data_recherche[feat] == value]
        else:
            num_feat.append(feat)

    ### On normalise les features numériques

    data_recherche_num = data_recherche.copy()
    data_recherche_num = data_recherche_num[num_feat]
    scaler = MinMaxScaler()
    scaler.fit(data_recherche_num)
    data_recherche_norm = pd.DataFrame(scaler.transform(data_recherche_num),index=data_recherche_num.index,columns=data_recherche_num.columns)
    actual_date_ts = pd.Timestamp(actual_date)
    Facte_ext_Cur  = scaler.transform(data.loc[[actual_date_ts],num_feat])
    data_cur_norm  = pd.Series(Facte_ext_Cur[0],index=num_feat)

    ### Calcul des distances entre la situation actuelle et l'historique
    
    # Facteurs_ext_historiques - Facteur_ext_courant

    data_recherche_norm_dist = data_recherche_norm[num_feat].sub(data_cur_norm, axis='columns')

    # Poids des facteurs
    
    for feat in data_recherche_norm_dist.columns:
        weight_ = weight_factors.loc[feat,'Poids Facteur %']
        data_recherche_norm_dist[feat] = weight_ * data_recherche_norm_dist[feat]


    # (Facteurs_ext_historiques - Facteur_ext_courant)**2

    data_recherche_norm_dist = data_recherche_norm_dist.applymap(lambda x: x**2)
    data_recherche_norm_dist['Proximite']     = data_recherche_norm_dist.sum(axis=1)
        
    data_recherche = pd.merge(data_recherche, data_recherche_norm_dist['Proximite'], left_index=True, right_index=True)
    
    data_recherche = data_recherche.sort_values(by='Proximite')

    
    return data_recherche.head(nbre_situations)


def Select_Historical_Context_Similar(data_hist=None,
                                      Variables = None,
                                      model = None,
                                      data_suivi = None,
                                      cur_date = None,
                                      epsilon = None,
                                      max_nhist_data_kept = None,
                                      min_nhist_data_required = None):

    import numpy as np

    nom_ipe_calcule = Variables['IPE_name']+'_Simule'
    

    data_suivi_cur     = data_suivi.loc[cur_date]     
    ipe_simule_current = data_suivi_cur[nom_ipe_calcule]

    data_hist = FilterCategory(Variables      = Variables, 
                               data_suivi_cur = data_suivi_cur, 
                               data_hist      = data_hist)



    data_historique_proches = data_hist.copy(deep=True)
   
    data_hist, data_suivi_cur = Normalize_FactExt(data_hist_filtered = data_hist, 
                                                  data_suivi_cur     = data_suivi_cur,
                                                  Variables          = Variables)

    data_hist = Compute_Distance (data_hist_normed      = data_hist, 
                                            data_suivi_cur_normed = data_suivi_cur,
                                            Variables             = Variables)

   

    data_hist['Proximite']               = data_hist.sum(axis=1)
    data_historique_proches['Proximite'] = data_hist['Proximite']

    # On tri les résultats par ordre de proximité
    data_historique_proches = data_historique_proches.sort_values(by=['Proximite'])
    data_historique_proches['Neighb_order'] = np.arange(1,len(data_historique_proches)+1)
    
    number_neighb_kept = Select_Number_Voisins(data_historique_proches=data_historique_proches, 
                                               Variables=Variables, 
                                               ipe_simule_current=ipe_simule_current, 
                                               epsilon=epsilon, 
                                               max_nhist_data_kept=max_nhist_data_kept,
                                               min_nhist_data_required=min_nhist_data_required)
                                 

    data_historique_proches_kept = data_historique_proches[data_historique_proches['Neighb_order']<=number_neighb_kept]
    
    return data_historique_proches_kept
    
    






def Predict_IPE(data        = [], 
                Variables   = [],
                model       = [], 
                start_date  = None, 
                end_date    = None):

    ''' Calcul de l'IPE en utilisant le modèle. 
        Si les données contiennent des variables catégorielles
        et que l'une des modalité n'est pas présente dans les données historique qui servies au modèle, celle-ci est remplacée 
        en utilisant le dictionnaire "SubstituteModality" contenu dans "Variables".

        *Parameters:

            data        : pandas DataFrame contenant les facteurs externes 
            Variables   : dict contenant la description des variables (Fext, Levier,Poids des facteurs)
            model       : sklearn model
            start_date  : date de début de la prévision
            end_date    : date de fin de la prévision

        *Retour:

            ts_predicted: Série temporelle Pandas avec la valeur prédite

    '''

    from copy import deepcopy

    Predicteurs = deepcopy(Variables['Fact_Ext_Num'])

    if 'Fact_Ext_Cat' in Variables.keys():

        Var_cat     = Variables['Fact_Ext_Cat']
        for vc in Var_cat.keys():
            Predicteurs.append(vc)

    if start_date == None:
        start_date = data.index[0]

    if end_date == None:
        end_date = data.index[-1]

    df_to_predict = data[Predicteurs].loc[start_date:end_date]


    if df_to_predict.empty:
        return None

    if 'Fact_Ext_Cat' in Variables.keys():

        for vc in Var_cat.keys():
            liste_modalite_suivi = list(df_to_predict[vc].unique())
            for l in liste_modalite_suivi:
                if l not in Var_cat[vc]:
                    value_substitute = Variables['SubstituteModality'][vc]
                    df_to_predict[vc] = df_to_predict[vc].map({l:value_substitute}).fillna(df_to_predict[vc])
            for l2 in Var_cat[vc]:
                if l2 not in liste_modalite_suivi:
                    df_to_predict[l2] = 0
    
    df_to_predict_dmy = pd.get_dummies(df_to_predict,prefix='',prefix_sep='')
    df_to_predict_dmy = df_to_predict_dmy[Variables['Fact_Ext_Glob']]

    #print(df_to_predict_dmy.columns.tolist())


    ts_predicted = pd.Series(index=df_to_predict.index, data=model.predict(df_to_predict_dmy))
    ts_predicted.name = Variables['IPE_name']+'_Simule'

    return ts_predicted


def FilterCategory(Variables      = None, 
                   data_suivi_cur = None, 
                   data_hist      = None):

    ''' Dans le cas où les données contiennent des variables catégorielles,
        filtrage des données historiques en ne conservant que les modalités de la situation actuelle.

        *Parameters:

            data_hist   : pandas DataFrame contenant les données historiques
            data_suivi  : pandas Serie contenant les données de la situation actuelle
            Variables   : dict contenant la description des variables (Fext, Levier,Poids des facteurs)


        *Retour:

            data_hist_filtered: pandas DataFrame contenant les données historiques filtrées

    '''

    filtre_cat = ''

    if 'Fact_Ext_Cat' in Variables.keys(): # Si il y'a des variables catégorielles

        Var_cat     = Variables['Fact_Ext_Cat']

        for vc in Var_cat.keys():
                cur_modalite_suivi = data_suivi_cur[vc]
                if cur_modalite_suivi not in Var_cat[vc]:
                    print('La modalité '+cur_modalite_suivi+ ' est non présente dans les données historiques')
                else:
                    filtre_cat = filtre_cat + ' & (data_hist[' + "'" + vc + "'" + ']=='+ "'" + cur_modalite_suivi + "'" + ")"
                
        filtre_categoriel = filtre_cat[3:]

        data_hist_filtered = data_hist[eval(filtre_categoriel)]

    else:

        data_hist_filtered = data_hist

    return data_hist_filtered


def Normalize_FactExt(data_hist_filtered = None, 
                      data_suivi_cur     = None,
                      Variables          = None):

    ''' Normalisation entre [0 1] des facteurs externes

        *Parameters:

            data_hist_filtered   : pandas DataFrame contenant les données historiques
            data_suivi_cur       : pandas Serie contenant les données de la situation actuelle
            Variables            : dict contenant la description des variables (Fext, Levier,Poids des facteurs)


        *Retour:

            data_hist_norm : pandas DataFrame contenant les données historiques normées
            data_cur_norm  : pandas Serie contenant les données de la situation actuelle normées
    '''


    from sklearn.preprocessing import MinMaxScaler

    Predicteurs = Variables['Fact_Ext_Num']
    data_hist_filtered = data_hist_filtered[Predicteurs] 

    scaler = MinMaxScaler()
    scaler.fit(data_hist_filtered)
    data_hist_norm = pd.DataFrame(scaler.transform(data_hist_filtered),index=data_hist_filtered.index,columns=data_hist_filtered.columns)
    Facte_ext_Cur    = scaler.transform(data_suivi_cur[Predicteurs].values.reshape(1, -1))
    data_cur_norm = pd.Series(Facte_ext_Cur[0],index=Predicteurs)

    return data_hist_norm, data_cur_norm


def Compute_Distance (data_hist_normed      = None, 
                      data_suivi_cur_normed     = None,
                      Variables          = None):

    ''' Normalisation entre [0 1] des facteurs externes

        *Parameters:

            data_hist_normed   : pandas DataFrame contenant les données historiques normées
            data_suivi_cur_normed       : pandas Serie contenant les données de la situation actuelle normées
            Variables            : dict contenant la description des variables (Fext, Levier,Poids des facteurs)


        *Retour:

            data_hist_normed : pandas DataFrame contenant les données historiques normées avec la Proximité
  
    '''
    Predicteurs = Variables['Fact_Ext_Num']

    # Facteurs_ext_historiques - Facteur_ext_courant
    data_hist_normed = data_hist_normed[Predicteurs].sub(data_suivi_cur_normed, axis='columns')

    # (Facteurs_ext_historiques - Facteur_ext_courant)**2
    data_hist_normed = data_hist_normed.applymap(lambda x: x**2)

    # Poids_Fact_Ext*(Facteurs_ext_historiques - Facteur_ext_courant)**2
    df_weight = pd.Series(Variables['WeightFactExtOnIPE']) 
    data_hist_normed = data_hist_normed*df_weight[Predicteurs]
    data_hist_normed['Proximite']     = data_hist_normed.sum(axis=1)
    
    return data_hist_normed



                                               

def Select_Number_Voisins(data_historique_proches=None,
                          Variables=None, 
                          ipe_simule_current=None, 
                          epsilon=None, 
                          max_nhist_data_kept=None,
                          min_nhist_data_required=None):

 
    nom_ipe_simule = Variables['IPE_name']+'_Simule'

    data_historique_proches['delta_IPE'] = data_historique_proches.apply(lambda x: abs((x[nom_ipe_simule]-ipe_simule_current)/x[nom_ipe_simule]), axis=1)

    df_filt = data_historique_proches[data_historique_proches['delta_IPE']>epsilon]


    kppv = df_filt.loc[df_filt.index[0], 'Neighb_order'] 
    
    return min(kppv,max_nhist_data_kept)