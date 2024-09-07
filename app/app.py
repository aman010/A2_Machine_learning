# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import flask
import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import Dash, html, Input, Output, callback, ALL, State, ctx
import copy


import plotly.graph_objects as go 
import pandas as pd
import plotly.express as ex
import matplotlib.pyplot as plt
from dash import Dash, dash_table
import numpy as np

from clean_data import clean_data


flask_app = flask.Flask(__name__)
dash_app = Dash(__name__, server = flask_app, external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=False,suppress_callback_exceptions=True)
df = pd.read_csv('data/Cars.csv')

df['mileage']=df['mileage'].apply(lambda x : str(x).split()[0]).astype('float')
df['engine'] = df['engine'].apply(lambda x: str(x).split()[0]).astype('float')
df['max_power'] = df['max_power'].apply(lambda x: str(x).split()[0])
df.drop(df[df['max_power'] == 'bhp'].index, inplace=True)
df['max_power'] = df['max_power'].astype('float')
df.drop(columns = 'torque', inplace=True)
df['name'] = df['name'].apply(lambda x:x.split()[0])
cols = df.columns.values
df['owner'].loc[df[(df['owner'] == 'Fourth & Above Owner') | (df['owner'] == 'Third Owner')].index] = 'others'
df.drop(df[df['owner']=='Test Drive Car'].index, inplace=True)
df_ = df.copy(deep=True)
df.drop(columns = {'selling_price'}, inplace=True)
df.drop(df[(df['fuel'] == 'LPG') | (df['fuel'] == 'CNG')].index, inplace=True)
max_models = df.groupby('name')['year'].max()
max_seat = df['seats'].max()
cols = df.columns
# ix = np.where(cols == 'selling_price')[0]
# cols = np.delete(cols, ix)


global _table 


dash_app.layout = html.Div(
    [
        # main app framework
        html.Div("Chakis Car Company", style={'fontSize':50, 'textAlign':'center'}),
        html.Div([
            html.A(page['name']+"  |  ", id={'index':page['name'], 'type':'childrenLinks'},
                   style={'cursor':'pointer'}, n_clicks=0)
            for page in dash.page_registry.values()
        ]),
        html.Hr(),
        html.Div(id='childrenContent', children=[],
                 style={'display':'flex', 'alignItems':'stretch','alignContent':'stretch', 'gap':'10px'}),
        
        
        
        html.Div( children = [
            html.Hr(style = {'width':'100%', 'height':'100%'}),
            
            html.Div(
                html.P("Welcome to veriosn 0.001 incubation portal for Cars Prediction. \
                       We adhere to the compliences provided and avoid data breaching. On to the left side of the screen  you can \
                       see two buttons ADD and SUBMIT .Once you click on the add button you created a store on the portal. \
                       Submit button will allow to submit the data for further Analytical Tasks.\
                       Data privacy is our utmost responsibility. The knowledge we gain from you is to provoke the better sales market for future."
                       
              , style = {'height':'100%', 'width':'100%', 'testAlign':'Right','bottom':'0%'})),  
                   ],
            style = {'position':'fixed','bottom': '0%', 'width':'70%','position':'fixed','left':'31%' }),

        html.Div(
            id='left-container', 
            style={'width':'30%', 'height':'100vh', 'float':'left', 'backgroundColor':'lightgray'},
            children=[html.Form([
                *[html.Div([
                    html.Label(col),
                    dcc.Dropdown(
                        id=col,
                        style = {'margin-bottom':'10px'},
                        options=[{'label': val, 'value': val} for val in df[col].unique()],
                        placeholder=df[col].iloc[0],
                        clearable = True,
                    ) if df[col].dtype == 'object' else 
                    dcc.Input(
                        type='number',
                        id=col,
                        style = {'margin-bottom':'10px', 'width':'98%', 'height':'80%'},
                        #value=df[col].iloc[0]
                        value=None
                    ), html.Div(id = 'em'+col, style = {'color':'red'}),
                        html.Div(id='notify'+col, style = {'color':'red'}),
                        html.Div(id = 'init'+col, style = {'color':'red'}),
                ]) for col in cols],
                html.Button('Add', id='submit', type ="button" , 
                            style = {'width':'50%', 'text-align': 'Center', 'display':'block', 'margin':'0 auto'}),
                    

                # dcc.Loading(id = "loading"),
           ]),
            
        ]),
        dcc.Store(id = "opt"),
        dcc.Store(id = "temp"),
        html.Div(id = "pred"),
        
        # html.Div(children= [
        #     dcc.Graph(id="res", style= {'display': 'inline-block'}),
        #     ], id = "table_res", 
        #     style = {'width':'50%', 'height': '15vh', 'float':'right', 'bottom':'30%'}),
        
        html.Div(children = [
            #dcc.Graph(id = 'op'),
            html.Div(children = [dcc.Graph(id = "gr", style = {"flex":'1', })], style = {'display':'flex',"height":"50%", "width":"50%"}),
            html.Div(children = [dcc.Graph(id = "res", style = {"flex":'2', })], style= {'display':'flex',"height":"50%", "width":"50%"}),
            
            html.Div(html.Button('Submit' ,id = 'clean', type='button'))
            ],
            
                    id = "table_output",
                    style={'display':'flex', "bottom":'65%'}         
                    #style={"height":'15vh','width':'50%',
                             #       'position':'fixed','bottom':'65%','margin':'0 auto','left':'30%','backgroundColor':'Gray'},
                             
                             ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Prediction portal")),
                dbc.ModalBody("Welcome To beta version of Car prediction portal, We are constantly Improving the prediction Rates with your feedback, \
                              The new beta version have added few validation on the input from user. upcoming new blend will help many car owners to validate the price \
                                  Down below you will find the instructions to follow."),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-button", className="ms-auto")
                ),
            ],
            id="popup-modal",
            is_open=True,  # Set to True to open on page load
        ),
    ])

#@app.callback(Output("output", 'children'),[Input("submit button", 'values')],
@dash_app.callback(
    Output('opt', 'data'),
    *[Output('init'+col, 'children') for col in cols],
    *[dash.dependencies.Output('em'+col, 'children') for col in cols],
    [Input('submit', 'n_clicks')],
    *[dash.dependencies.State(col, 'value') for col in cols],
    *[dash.dependencies.State('em'+col, 'children') for col in cols],
    preventDefault=True,prevent_initial_call=True)

def process_form(n_clicks, *values):
    if n_clicks is not None:
        #perform some cleaning if required 
        print("processing store")
        em = np.repeat(None, cols.shape[0])
        if np.array([values]).any() == None:
            hb = []
            for i in range(cols.shape[0]):
                hb.append(html.Div( style = {'backgroundColor':'red', 'height':'20px','width':'20px','margin':'5px', 'border':'1px solid red'}
                    ))
            return None,*em,*hb
        else:
            df_ = pd.DataFrame(dict(zip(cols , values)),index=(len(values),))
            df_.fillna(value=np.nan)
            # if em.any():
                #     em = np.repeat(None, cols.shape[0])
            df_.seats.astype('float')
            if df_[df_.seats > df['seats'].max()].shape[0] >= 1:
                    #em=process_form(n_clicks, values)
                    em[10] = 'Number of seats Must not be grater than {}'.format(df['seats'].max())
            df_.engine.astype('float')
            if df_[df_['engine'] > 2500].shape[0] >= 1:
                em[8]= 'The engine in CC should only be upto 2500'
        
            #maximum year of specific car
            df.year.astype('float')
            max_ = max_models[values[0]]
            if df_[df_['year'] > max_].shape[0] >= 1:
                em[1] = 'This year car model not found'
            
        # if df_[df_['name'] == None].shape[0] >= 1:
        #     em[0] = "Please Select the Car"        
        return values,*em, *np.repeat(None, cols.shape[0])

 #prevent_initial_call=True,)
@dash_app.callback([Output('gr', 'figure'),Output('res', 'figure')],
               Output('temp', 'data'),*[Output(col , 'value') for col in cols],
               *[dash.dependencies.Output('notify'+col, 'children') for col in cols],
              [Input('opt', 'data'), Input('temp', 'data')],
              dash.dependencies.Input('submit', 'n_clicks'),dash.dependencies.Input('clean', 'n_clicks'),
              *[dash.dependencies.Input('em'+col, 'children') for col in cols],
              preventDefault=True,prevent_initial_call=True)
#*[dash.dependencies.State(col, 'value') for col in df.columns])

def table_processing(data,temp,n_clicks, n_clicks2, *em):
    if np.array(data).any():
        print("processing data")
        print(data)
        if temp is None:
            #ccheck the data is not null before adding to the store
            df_ = pd.DataFrame(dict(zip(cols , data)),index=(len(data),))
            #update the opt store with current value
            temp =[]
            temp.append(data)
        else:
            temp.append(data)
            index = list(range(np.array(temp).shape[0])) 
            # index = list(range(n_clicks))
            df_ = pd.DataFrame(dict(zip(cols, np.array(temp).T)), index = index)
            
        ctx = dash.callback_context       
        
        fig2 = go.Figure(data= [go.Table(header = dict(values= ['prediction','blend']))])
        
        fig = go.Figure(data=[go.Table(header=dict(values=list(cols)))])
        figz = None
        
        notify = None
        
        #if np.array(em).any() == None:
        print('another context is created')
        if ctx.triggered:
            prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
            print(prop_id)
            if prop_id == 'submit':
                print('the df',df_)

                # fig = go.Figure(data=[go.Table(header=dict(values=list(cols)),
                #                                cells=dict(values=list(df_.values.T))
                #   
                #again chedk for each variable 
                #if np.array(notify).any() == None:
                # em=np.array(em)
                max_ = max_models[data[0]]
                #max_ = temp[len(temp)-1][0]
                
                # table = fig.data[0]
                # table.cells.values = list(df_.values.T)
                #em = np.repeat(None, cols.shape[0])  
                     
                print(fig)
                     
                Output_values = {'gr':fig,'res':None,'temp':temp}
                    #fig.update_layout(width= 800, margin='0 auto',)
                     
                if df_[df_['seats'] > 14].shape[0] >= 1:
                    #em=process_form(n_clicks, values)
                    #update the last state of store
                    idx=len(temp)
                    temp = temp[:-1]
                    em = np.array(em)
                    notify = 'Please Enter the approprate number of seats'
                    #update the current temp state
                    em[10]= notify
                    figz = {}, {'loading_state':True, 'component_name':'gr'}, 
                    style={'height':'400px','width':'800px','display':'flex', 'justifyContent':'center', 'alignitemr':'center'}
                    return figz, fig2, temp, dash.no_update,*em
                   

                if df_[df_['engine'] > 2500].shape[0] >= 1:
                    
                    notiy  = 'The engine in CC should only be upto 2500'
                    temp = temp[:-1]
                    em = np.array(em)
                    em[8] = notify
                    figz = {}, {'loading_state':True, 'component_name':'gr'}, 
                    style={'height':'400px','width':'800px','display':'flex', 'justifyContent':'center', 'alignitemr':'center'}
                    return figz, fig2, dash.no_update,temp, *em
                    
                if df_[df_['year'] > max_].shape[0] >= 1:
                    
                    idx = len(temp)
                    temp = temp[:-1]
                    em = np.array()
                    notyify = 'This year car model not found'
                    print(temp[idx - 1][0])

                    em[1] = notify
                    figz = {}, {'loading_state':True, 'component_name':'gr'}, 
                    style={'height':'400px','width':'800px','display':'flex', 'justifyContent':'center', 'alignitemr':'center'}
                    fig3 = go.Figure()
                    

                    
                    return figz, fig2, temp, dash.no_update,*em
                
                print('im here')
                table = fig.data[0]
                table.cells.values = list(df_.values.T)
                notify = np.repeat(None, cols.shape[0])
                df_ = pd.DataFrame(columns = cols)
                print('before returning' ,Output_values.values())  
                #empty the form return
                #print('***************************', vals)
                #vals = []
                up_f = np.repeat(np.NAN, cols.shape[0])
                return fig, fig2, temp, *up_f, *em,
        
            if prop_id == 'clean':
                index = list(range(np.array(temp).shape[0]))
                print('index in clean', index)
                print("cleaning process")
                clean = clean_data(x_train_data=df[cols],data = np.array(temp))
                
                p = clean._clean()
                if p is None:
                    print('Not enough data')
                print('the cleaned data', p)
                print('performing predictions')
                pred, blend = clean.perfom_prediction(p)
                print('res of pred', (pred, blend))
                l = ['prediction', 'blended prediction']
                
                # fig2 = go.Figure(data=[go.Table(header=dict(values=l),
                #                        cells=dict(values=np.array([pred[0],blend[0]]).T))
                #                        ])
                
                table = fig2.data[0]
                if pred.shape[0] == 1:
                    
                    table.cells.values = np.array([pred[0],blend[0]]).T
                else:
                    table.cells.values = np.array([pred,blend]).T
                    
                
                Output_values = {'gr':fig2,
                                'res':fig2,
                                'temp':None}
                
        
                print('before returning' ,Output_values.values()) 
                
                
                up_f = np.repeat(np.NAN, cols.shape[0])
                return fig, fig2, temp, *up_f,*em
    else:
       raise dash.exceptions.PreventUpdate()
       
       
@dash_app.callback(
    Output("popup-modal", "is_open"),
    [Input("close-button", "n_clicks")],
    [State("popup-modal", "is_open")],
)
def toggle_modal(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open    
    
# @app.callback(Output('test_output', 'children', allow_duplicate=True),
#               [Input('clean', 'n_clicks')],
# prevent_initial_call=True)
# def data_processing(n_clicks, temp):
#     print('inside clean')
#     print('cleaning process starts')
#     fig = px.line(x = [1,2,3], y = [1,2,3])
#     return dcc.Graph(id = 'ap',figure = fig)

    
    '''table = html.Table([
           ([(col) for col in df_.columns]),
           html.Tbody([
               ([
                 (df_.iloc[i][col]) for col in df_.columns
               ]) for i in range(len(df_))
           ])
       ])
    
        
        print(df_)
        print(table)
        return table
        return html.Table([
            [(col) for col in df.columns],
            html.Tbody([
                [df.iloc[i][col] for col in df.columns] 
            for i in range(len(df)) ]) ])'''
            


if __name__ == "__main__":
    dash_app.run_server(debug=False, host='0.0.0.0',port=8089)
