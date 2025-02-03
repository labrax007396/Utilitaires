def plot_model_mesure(model,data):
    
    import plotly.graph_objects as go

    Y  = data[data.columns[0]]
    X  = data.drop(columns=data.columns[0])
    Y_pred = model.predict(X)
    import pandas as pd
    df_ = pd.DataFrame(index=data.index,data={'Mesure':Y.values,'Model':Y_pred})

    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_.index, y=df_.Mesure,
                        mode='lines',
                        name='Mesure'))
    fig.add_trace(go.Scatter(x=df_.index, y=df_.Model,
                        mode='lines',
                        name='Modèle'))

    fig.show()

def plot_model_vs_mesure(model,data):
    
    import plotly.graph_objects as go

    Y  = data[data.columns[0]]
    X  = data.drop(columns=data.columns[0])
    Y_pred = model.predict(X)
   

    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Y.values, y=Y_pred,
                        mode='markers',
                        name='Mesure .vs. Modele'))
 
    fig.show()
