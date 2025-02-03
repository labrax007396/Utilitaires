from scipy.stats import t
from scipy import sqrt
from statistics import variance, mean 
import pandas as pd

def int_ech(values,conf=0.95) :
    n = len(values) 
    m = mean(values) 

    s = variance(values)


    proba = (1-conf)*100 ; proba = (100-proba/2)/100 

    ddl = n - 1

    intervalle = sqrt(s) * t.ppf(proba, ddl)

    return intervalle


def calcul_interv_conf(data:pd.DataFrame, conf_level:float, seuil_min:float):        

    data_cpy = data[data.Mesure>seuil_min]
    residu = (data_cpy['Mesure']-data_cpy['Modele'])
    residu = residu.dropna()

    if len(residu)<2:
        print("Nombre de valeurs insuffisant pour calculer l'intervalle de confiance")
        return None
    else:
        ic = int_ech(residu.values,conf=conf_level)
        return ic 