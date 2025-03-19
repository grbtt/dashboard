import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
from sklearn import  metrics
import numpy as np
import joblib 

#Define CSS style
external_stylesheets = ["assets/style.css"]


#Load data
df_raw = pd.read_csv('testData_2019_Civil.csv')

df_raw['Date'] = pd.to_datetime (df_raw['Date']) # create a new column 'data time' of datetime type



df_raw['Hour']=df_raw['Date'].dt.hour
df_raw['power-1']=df_raw["Civil (kWh)"].shift(1)
df_raw = df_raw.dropna()
print(df_raw.head())

df_real = df_raw.iloc[:,0:2]
print(df_real.head())

df = df_raw.drop (columns = ["Civil (kWh)",'HR','windSpeed_m/s', 'windGust_m/s','pres_mbar','rain_mm/h','rain_day']) 
print(df.head())


y2=df_real["Civil (kWh)"].values

df2=df.iloc[:,1:5]
print(df2.head())

X2=df2.values
fig1 = px.line(df, x="Date", y=df.columns[1:5])# Creates a figure with the raw data

#feature selection
df3= df_raw.drop(columns = ['Date']) 
available_columns = df_raw.columns[2:].tolist()
Z=df3.values

Y=Z[:,0]
X=Z[:,[1,2,3,4,5,6,7,8,9,10]] 
print(Y)
print(X)


from sklearn.feature_selection import SelectKBest # selection method
from sklearn.feature_selection import mutual_info_regression,f_regression # score metric (f_regression)

# With 3 features

#Create a variable using selectkbest with k features and f_regression  
features=SelectKBest(k=3,score_func=f_regression)  

# Compute the correlation between features and output
fit=features.fit(X,Y)                                                                                
features_results=fit.transform(X)                           
features = df3.columns[1:].tolist()

# Création d'un DataFrame
df_scores = pd.DataFrame({'Feature': features, 'Score': fit.scores_})

# ensemble method 
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X,Y)
# Création d'un DataFrame
df_importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})



# Modèle Random Forest
RF_model = joblib.load('RF_model.sav')

y2_pred_RF = RF_model.predict(X2)

#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y2,y2_pred_RF)
MBE_RF=np.mean(y2-y2_pred_RF) 
MSE_RF=metrics.mean_squared_error(y2,y2_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y2,y2_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y2)
NMBE_RF=MBE_RF/np.mean(y2)

# Modèle Neural Network 
NN_model = joblib.load('NN_model.sav')

y2_pred_NN = NN_model.predict(X2)

#Evaluate errors
MAE_NN=metrics.mean_absolute_error(y2,y2_pred_NN)
MBE_NN=np.mean(y2-y2_pred_NN) 
MSE_NN=metrics.mean_squared_error(y2,y2_pred_NN)  
RMSE_NN= np.sqrt(metrics.mean_squared_error(y2,y2_pred_NN))
cvRMSE_NN=RMSE_NN/np.mean(y2)
NMBE_NN=MBE_NN/np.mean(y2)

# Modèle Linear Regression

LR_model = joblib.load('LR_model.sav')

y2_pred_LR = LR_model.predict(X2)

#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y2,y2_pred_LR) 
MBE_LR=np.mean(y2-y2_pred_LR)
MSE_LR=metrics.mean_squared_error(y2,y2_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y2,y2_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y2)
NMBE_LR=MBE_LR/np.mean(y2)



# Create data frames with predictin results and error metrics 
d_metrics = {'Methods': ['Linear Regression','Random Forest','NeuralNetwork'], 'MAE': [MAE_LR, MAE_RF, MAE_NN],'MBE': [MBE_LR, MBE_RF, MBE_NN], 'MSE': [MSE_LR, MSE_RF, MSE_NN], 'RMSE': [RMSE_LR, RMSE_RF, RMSE_NN],'cvMSE': [cvRMSE_LR, cvRMSE_RF, cvRMSE_NN],'NMBE': [NMBE_LR, NMBE_RF, NMBE_NN]}
df_metrics = pd.DataFrame(data=d_metrics)
d={'Date':df['Date'].values, 'LinearRegression': y2_pred_LR}
LR_forecast=pd.DataFrame(data=d)
d={'Date':df['Date'].values, 'RandomForest': y2_pred_RF}
RF_forecast=pd.DataFrame(data=d)
d={'Date':df['Date'].values, 'NeuralNetwork': y2_pred_NN}
NN_forecast=pd.DataFrame(data=d)

# merge real and forecast results and creates a figure with it
LR_results=pd.merge(df_real,LR_forecast, on='Date')
RF_results=pd.merge(df_real,RF_forecast, on='Date')
NN_results=pd.merge(df_real,NN_forecast, on='Date')



# Define auxiliary functions
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    html.H1('IST Energy Forecast tool (kWh)'),
    html.P('Representing Data, Forecasting and error metrics for year 2019 using three tabs'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
        dcc.Tab(label='Error Metrics', value='tab-3'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H4('IST Raw Data'),
            html.H4('Choose variables you want to plot :'),
            dcc.Dropdown(
                id='column-selector',
                options=[{'label': col, 'value': col} for col in available_columns],
                value=[available_columns[0]],  # Valeur par défaut
                multi=True  # Permet de sélectionner plusieurs colonnes
            ),
            dcc.RadioItems(
                id='graph-type',
                options=[
                    {'label': 'Line graph', 'value': 'line'},
                    {'label': 'Boxplot', 'value': 'box'}
                ],
                value='line',  # Valeur par défaut
                inline=True  # Affichage horizontal des options
            ),
            html.Div(id='graph-container'),
            html.H4('Choose feature selection method :'),
            dcc.RadioItems(
                id='feature-type',
                options=[
                    {'label': 'Ensemble Method', 'value': 'ensemble'},
                    {'label': 'KBest Method', 'value': 'kbest'}
                ],
                value='ensemble',  # Valeur par défaut
                inline=True  # Affichage horizontal des options
            ),
            dcc.Graph(id='dynamic-feature')
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H4('IST Electricity Forecast (kWh)'),
            html.H4('Choose regression method :'),
            dcc.RadioItems(
                id='regression-type',
                options=[
                    {'label': 'Linear Regression', 'value': 'LR'},
                    {'label': 'Neural Network', 'value': 'NN'},
                    {'label': 'Random Forest', 'value': 'RF'}
                ],
                value='RF',  # Valeur par défaut
                inline=True  # Affichage horizontal des options
            ),
            dcc.Graph(id='dynamic-graph')
            
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H4('IST Electricity Forecast Error Metrics'),
            dcc.Dropdown(
                id='metrics-selector',
                options=[{'label': metric, 'value': metric} for metric in ['MAE','MBE', 'MSE', 'RMSE','cvMSE','NMBE']],
                value='MAE',  
                multi=True  
            ),
            html.Div(id='metrics-table')
        ])

@app.callback(
    Output('graph-container', 'children'),
    [Input('column-selector', 'value'),
     Input('graph-type', 'value')]
)
def update_graphs(selected_columns, graph_type):
    graphs = []

    if graph_type == 'line':
        fig = px.line(df_raw, x="Date", y=selected_columns, title="Selected variables")
        graphs.append(dcc.Graph(figure=fig))
    
    else:
        for col in selected_columns:
            fig = px.box(df_raw, y=col, title=f"Boxplot of {col}")
            graphs.append(dcc.Graph(figure=fig))

    return graphs

@app.callback(    
    Output('dynamic-feature', 'figure'),
    Input('feature-type', 'value'))
def update_feature(method):
    if method == 'kbest':
        fig = px.bar(df_scores, x='Feature', y='Score', title='Feature to select - F Regression')
    if method == 'ensemble':
        fig = px.bar(df_importance, x='Feature', y='Importance', 
             title='Features to Select - RANDOM FOREST',
             labels={'Importance': 'Feature Importance'})
    return fig

@app.callback(    
    Output('dynamic-graph', 'figure'),
    Input('regression-type', 'value'))
def update_graph(reg):
    if reg == 'RF':
        fig2 = px.line(RF_results,x=RF_results.columns[0],y=RF_results.columns[1:3])
    if reg == 'LR':
        fig2 = px.line(LR_results,x=LR_results.columns[0],y=LR_results.columns[1:3])
    if reg == 'NN':
        fig2 = px.line(NN_results,x=NN_results.columns[0],y=NN_results.columns[1:3])
    return fig2

@app.callback(    
    Output('metrics-table', 'children'),
    Input('metrics-selector', 'value'))
def update_table(selected_metrics):
    d = {'Methods': ['Linear Regression','Random Forest','Neural Network']}
    for m in selected_metrics:
        d[m]=d_metrics[m]
    df_metrics = pd.DataFrame(data=d)
    return generate_table(df_metrics)


if __name__ == '__main__':
    app.run(debug=False)
