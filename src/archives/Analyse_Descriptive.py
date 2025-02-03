import pandas as pd

# Valeurs manquantes #

def MissingValues(x: pd.DataFrame):
    import missingno as msno
    df_missing = x.apply(num_missing, axis=0)
        
    print("% de valeurs manquantes par colonne:")
    print(df_missing)
    fig = msno.matrix(x,figsize=(10,5), fontsize=12)
    return fig.get_figure()

def num_missing(x):
    perc_missing = round(100*sum(x.isnull())/len(x),1)
    return perc_missing

# Histogrammes #

def histo(data: pd.DataFrame, show_stat_on_unused_features:bool, dico_model:dict):
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if show_stat_on_unused_features:
        var_numerique_desc  = [dico_model['description']] + [f['description'] for f in dico_model['facteurs'].values() if f['type']=='num']
        var_numerique_alias = [dico_model['tag_name']] + [f['nom'] for f in dico_model['facteurs'].values() if f['type']=='num']
        list_unit           = [dico_model['tag_unit']] + [f['unit'] for f in dico_model['facteurs'].values() if f['type']=='num']
    else:
        var_numerique_desc  = [dico_model['description']] + [f['description'] for f in dico_model['facteurs'].values() if f['type']=='num' and f['used']]
        var_numerique_alias = [dico_model['tag_name']] + [f['nom'] for f in dico_model['facteurs'].values() if f['type']=='num' and f['used']]
        list_unit           = [dico_model['tag_unit']] + [f['unit'] for f in dico_model['facteurs'].values() if f['type']=='num' and f['used']]



    #var_numerique_unit = [v+' '+u for v,u in zip(var_numerique_desc, list_unit)]

    data_num = data[var_numerique_alias]

    n_var_num = len(var_numerique_alias)

    if n_var_num%2 == 0:
        nrows = int(n_var_num/2)
    else:
        nrows = int((n_var_num+1)/2)

    fig_histo = make_subplots(rows=nrows, cols=2,subplot_titles=tuple(var_numerique_desc))

    if n_var_num%2 == 0:
        for row in range(nrows):
            fig_histo.add_trace(go.Histogram(x=data_num[var_numerique_alias[2*row]]),row=row+1, col=1)
            fig_histo.add_trace(go.Histogram(x=data_num[var_numerique_alias[2*row+1]]),row=row+1, col=2)
    else:
        for row in range(nrows-1):
            fig_histo.add_trace(go.Histogram(x=data_num[var_numerique_alias[2*row]]),row=row+1, col=1)
            fig_histo.add_trace(go.Histogram(x=data_num[var_numerique_alias[2*row+1]]),row=row+1, col=2)

        fig_histo.add_trace(go.Histogram(x=data_num[var_numerique_alias[2*nrows-2]]),row=nrows, col=1)
        

    fig_histo.update_annotations(font_size=12)
    fig_histo.update_layout(
        title_text="Histogramme des variables numériques",
        autosize=False,
        width=1000,
        height=nrows*300)

    for i, unit in enumerate(list_unit): 
        fig_histo['layout']['xaxis{}'.format(i+1)]['title']=unit



    fig_histo.show()



    if show_stat_on_unused_features:
        var_disc_desc  = [f['description'] for f in dico_model['facteurs'].values() if f['type']=='cat']
        var_disc_alias = [f['nom'] for f in dico_model['facteurs'].values() if f['type']=='cat']
        list_unit      = [f['unit'] for f in dico_model['facteurs'].values() if f['type']=='cat']
    else:
        var_disc_desc  = [f['description'] for f in dico_model['facteurs'].values() if f['type']=='cat' and f['used']]
        var_disc_alias = [f['nom'] for f in dico_model['facteurs'].values() if f['type']=='cat' and f['used']]
        list_unit      = [f['unit'] for f in dico_model['facteurs'].values() if f['type']=='cat' and f['used']]

    if len(var_disc_alias)>0:


        liste_df_var_cat = list()

        for var_cat in var_disc_alias:
            df_perc_per_cat = 100*data.groupby([var_cat])[var_cat].count()/len(data)
            df_perc_per_cat = df_perc_per_cat.sort_values(ascending = False)
            liste_df_var_cat.append(df_perc_per_cat)


        n_var_cat = len(var_disc_alias)

        fig = make_subplots(rows=n_var_cat, cols=1,subplot_titles=tuple(var_disc_desc))


        for row,df in enumerate(liste_df_var_cat):
            fig.add_trace(go.Bar(y=df.values,x=df.index.to_list()),row=row+1, col=1)
            fig.update_layout(xaxis_tickangle=-45)


            

        fig.update_annotations(font_size=12)
        fig.update_layout(
            title_text="Répartition des variables discrètes",
            autosize=False,
            width=800,
            height=nrows*200)

        for i, unit in enumerate(var_disc_alias): 
            fig['layout']['yaxis{}'.format(i+1)]['title']="Occurence (%)"

        fig.show()

    else:
        fig = None

    return fig_histo, fig


# Box plots #

def box(data: pd.DataFrame, show_stat_on_unused_features:bool, dico_model:dict):

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots



    if show_stat_on_unused_features:
        var_numerique_desc  = [dico_model['description']] + [f['description'] for f in dico_model['facteurs'].values() if f['type']=='num']
        var_numerique_alias = [dico_model['tag_name']] + [f['nom'] for f in dico_model['facteurs'].values() if f['type']=='num']
        list_unit           = [dico_model['tag_unit']] + [f['unit'] for f in dico_model['facteurs'].values() if f['type']=='num']
    else:
        var_numerique_desc  = [dico_model['description']] + [f['description'] for f in dico_model['facteurs'].values() if f['type']=='num' and f['used']]
        var_numerique_alias = [dico_model['tag_name']] + [f['nom'] for f in dico_model['facteurs'].values() if f['type']=='num' and f['used']]
        list_unit           = [dico_model['tag_unit']] + [f['unit'] for f in dico_model['facteurs'].values() if f['type']=='num' and f['used']]




    #var_numerique = [v for v in data.columns if data[v].dtypes != 'object']
    data_num = data[var_numerique_alias]

    n_var_num = len(var_numerique_alias)

    if n_var_num%2 == 0:
        nrows = int(n_var_num/2)
    else:
        nrows = int((n_var_num+1)/2)

    fig = make_subplots(rows=nrows, cols=2,subplot_titles=tuple(var_numerique_desc))

    if n_var_num%2 == 0:
        for row in range(nrows):

            b_plot_1 = go.Box(
                y=data_num[var_numerique_alias[2*row]],
                boxpoints='outliers'
            )
            b_plot_2 = go.Box(
                y=data_num[var_numerique_alias[2*row+1]],
                boxpoints='outliers'
            )

            fig.add_trace(b_plot_1,row=row+1, col=1)
            fig.add_trace(b_plot_2,row=row+1, col=2)
    else:
        for row in range(nrows-1):

            b_plot_1 = go.Box(
                y=data_num[var_numerique_alias[2*row]],
                boxpoints='outliers' # only outliers

            )
            b_plot_2 = go.Box(
                y=data_num[var_numerique_alias[2*row+1]],
                boxpoints='outliers'
            )

            fig.add_trace(b_plot_1,row=row+1, col=1)
            fig.add_trace(b_plot_2,row=row+1, col=2)

        b_plot_3 = go.Box(
                    y=data_num[var_numerique_alias[2*nrows-2]],
                    boxpoints='outliers'
        )

        fig.add_trace(b_plot_3,row=nrows, col=1)
        

    fig.update_annotations(font_size=12)
    fig.update_layout(
        title_text="Box plot des variables numériques",
        autosize=False,
        width=1000,
        height=nrows*300)


    for i, unit in enumerate(list_unit): 
        fig['layout']['yaxis{}'.format(i+1)]['title']=unit
    #    fig['layout']['xaxis{}'.format(i+1)]['title']=''


    fig.show()

    return fig   


def correlation(data: pd.DataFrame):

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go


    var_numerique = [v for v in data.columns if data[v].dtypes != 'object']
    data_num = data[var_numerique]

    n_var_num = len(var_numerique)


    if n_var_num%2 == 0:
        nrows = int(n_var_num/2)
    else:
        nrows = int((n_var_num+1)/2)

    fig = make_subplots(rows=nrows, cols=2, subplot_titles=tuple(var_numerique[1:]))

    if n_var_num%2 == 0:
        for row in range(nrows-1):
            fig.add_trace(go.Scatter(x=data_num[var_numerique[2*row+1]], y=data_num[var_numerique[0]],mode='markers',showlegend=False),row=row+1, col=1)
            fig.add_trace(go.Scatter(x=data_num[var_numerique[2*row+2]], y=data_num[var_numerique[0]],mode='markers',showlegend=False),row=row+1, col=2)
            fig.update_yaxes(title_text=var_numerique[0], row=row+1, col=1)
            fig.update_yaxes(title_text=var_numerique[0], row=row+1, col=2)

        fig.add_trace(go.Scatter(x=data_num[var_numerique[n_var_num-1]], y=data_num[var_numerique[0]],mode='markers',showlegend=False),row=int(n_var_num/2), col=1)
        fig.update_yaxes(title_text=var_numerique[0], row=int(n_var_num/2), col=1)
        

    else:
        for row in range(nrows-1):
            fig.add_trace(go.Scatter(x=data_num[var_numerique[2*row+1]], y=data_num[var_numerique[0]],mode='markers',showlegend=False),row=row+1, col=1)
            fig.add_trace(go.Scatter(x=data_num[var_numerique[2*row+2]], y=data_num[var_numerique[0]],mode='markers',showlegend=False),row=row+1, col=2)

            #fig.update_xaxes(title_text=var_numerique[2*row+1], row=row+1, col=1)
            fig.update_yaxes(title_text=var_numerique[0], row=row+1, col=1)
            #fig.update_xaxes(title_text=var_numerique[2*row+2], row=row+1, col=2)
                

    fig.update_annotations(font_size=12)
    fig.update_layout(
        title_text="Visualisation des corrélations entre variable modélisée et facteurs",
        autosize=False,
        font=dict(
            family="Calibri",
            size=12
        ),
        width=1000,
        height=nrows*400)


    fig.show()
    return fig


def correlogramme(data: pd.DataFrame):
        
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sn

    fig_size = data.shape[1]
    plt.figure(figsize = (fig_size,fig_size))
    corrMatrix = data.corr()

    ax = plt.axes()
    mask = np.triu(np.ones_like(corrMatrix))

    plot_coefcor = sn.heatmap(corrMatrix, annot=True,vmin=-1, vmax=1, center=0,mask=mask,
                            cmap=sn.diverging_palette(20, 220, n=200),
                            square=True, ax = ax)
    ax.set_title('Coefficients de corrélation')
    #plt.show()

    fig = plot_coefcor.get_figure()
    
    return fig