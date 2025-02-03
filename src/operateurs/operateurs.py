
import datetime as dt
from datetime import timedelta
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from scipy import interpolate



def mean_value_on_new_index(df1,df2):



    df1['epoch'] = (df1.index - dt.datetime(1970,1,1)).total_seconds()
    df2['date_debut'] = (df2.index - dt.datetime(1970,1,1)).total_seconds()
    df2['date_fin'] = df2.apply(lambda row: row['date_debut'] + 60*row['duree'],axis=1)

    f = interpolate.interp1d(df1['epoch'].values.flatten(), df1['value'].values.flatten(),bounds_error=False)

    df2['value'] = df2.apply(Calcul_conso,f=f,axis=1)

    return df2
    

def Calcul_conso(raw,f):


    f_interp = 60.0 # secondes

    delta_time = raw['date_fin']-raw['date_debut']
    if delta_time < f_interp:
        value_aggregee = 0.0
    else:
        nsubdiv = round(delta_time/f_interp)+1
        new_index = np.arange(raw['date_debut'],raw['date_fin']+(raw['date_fin']-raw['date_debut'])/nsubdiv,(raw['date_fin']-raw['date_debut'])/nsubdiv)
        value_interpolee = f(new_index)
        value_aggregee = np.mean(value_interpolee)
    return value_aggregee

