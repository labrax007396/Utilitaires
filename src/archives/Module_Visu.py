def show_ts_data(**kwargs):

   
    """ Visualisation de times series de 2 jeux de données pandas 
        avec éventuellement double échelle en y.

        Parameters:

        data_1: pandas data frame
        data_2: pandas data frame
        vars_1: colonnes de data_1 à visualiser
        vars_2: colonnes de data_2 à visualiser
        style_1: style du graphe data_1 ('lines'/'markers'/'lines+markers')
        style_2: style du graphe data_2 ('lines'/'markers'/'lines+markers')
        size: taille de la figure : [dim1,dim2]
        secondary_y: True/False
        
        Returns:
        None

   """




    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data_1 = kwargs['data_1']
    vars_1 = kwargs['vars_1']
    if 'style_1' in kwargs.keys():
        style_1 = kwargs['style_1']
    else:
        style_1 = 'lines'

    if 'style_2' in kwargs.keys():
        style_2 = kwargs['style_2']
    else:
        style_2 = 'lines'

    


    if 'data_2' in kwargs.keys() and 'secondary_y' in kwargs.keys():

        data_2 = kwargs['data_2']
        vars_2 = kwargs['vars_2']

        if 'leg_2' in kwargs.keys():
            leg_2 = kwargs['leg_2']
        else:
            leg_2 = vars_2

        if 'leg_1' in kwargs.keys():
            leg_1 = kwargs['leg_1']
        else:
            leg_1 = vars_1




        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for v, l in zip(vars_1, leg_1):
            fig.add_trace(go.Scatter(x=data_1.index, y=data_1[v],
                        mode=style_1,
                        name=l),secondary_y = False)

        for v, l in zip(vars_2, leg_2):
            fig.add_trace(go.Scatter(x=data_2.index, y=data_2[v],
                        mode=style_2,
                        name=l),secondary_y=kwargs['secondary_y'])


    elif 'data_2' in kwargs.keys() and 'secondary_y' not in kwargs.keys():

        data_2 = kwargs['data_2']
        vars_2 = kwargs['vars_2']

        fig = go.Figure()

        for v in vars_1:
            fig.add_trace(go.Scatter(x=data_1.index, y=data_1[v],
                        mode=style_1,
                        name=v))

        for v in vars_2:
            fig.add_trace(go.Scatter(x=data_2.index, y=data_2[v],
                        mode=style_2,
                        name=v))

    else:

        fig = go.Figure()

        for v in vars_1:
            fig.add_trace(go.Scatter(x=data_1.index, y=data_1[v],
                        mode=style_1,
                        name=v))


    if 'size' in kwargs.keys():
        dim = kwargs['size']
        fig.update_layout(
                            autosize=False,
                            width=dim[0],
                            height=dim[1],
                            legend=dict(
                                        yanchor="top",
                                        y=0.99,
                                        xanchor="left",
                                        x=0.01
                                        )
                          )

                            

    fig.show()

def ModelPerf(y_train, y_predicted_train, y_test, y_predicted_test, R2_appr,R2_test,MAPE,title ):

    import matplotlib.pyplot as plt
       

    fig = plt.figure(1, figsize=(6, 6))

    ax = plt.gca()
    ax.set_xlabel('Mesure')
    ax.set_ylabel('Modèle')
    ax.set_title(title)
    plt.scatter(y_train.values, y_predicted_train, marker= 'o', color='b',label='Apprentissage')
    plt.scatter(y_test.values, y_predicted_test, marker= 'o', color='g',label='Validation')
    plt.plot([0,y_train.values.max()], [0,y_train.values.max()], 'r-')
    ax.text(0.95, 0.19, '$R^2$' + " Apprent = " + str(round(R2_appr,3)),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=12)
    ax.text(0.95, 0.12, '$R^2$' + " Test = " + str(round(R2_test,3)),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=12)
    ax.text(0.95, 0.05, "MAPE = "+str(round(100*MAPE,1)) + "%",
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=12)   

    plt.xlim(left=0, right=y_train.values.max())
    plt.ylim(bottom=0, top=y_train.values.max())



    plt.legend()

    plt.show()    