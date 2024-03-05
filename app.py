from dash import Dash
from dash import dcc, html
from dash.dependencies import Output, Input, State

from dash_bootstrap_templates import load_figure_template
import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import itertools
import os
from textwrap import wrap
import base64
import io

from wordcloud import WordCloud
from deep_translator import GoogleTranslator
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# import webbrowser


from share_functions import *
from word_cloud import *

# # Cheetsheet: https://dashcheatsheet.pythonanywhere.com/

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

vis = pd.read_csv("./data/vis_5nd.csv")
act = pd.read_csv("./data/act_5nd.csv")
ans = pd.read_csv("./data/ans_5nd.csv")
cart = pd.read_csv("./data/cart_5nd.csv")

act['timestamp'] = pd.to_datetime(act['timestamp'])
act['product_id'] = act['product_id'].apply(lambda x: int(float(x)) if x != '-' else x)
cart['product_id'] = cart['product_id'].apply(lambda x: int(float(x)) if x != '-' else x)
ans['question'] = ans['question'].apply(lambda x: x.replace('.', ' '))

def fix_long_prod_names(row):
    if row['product_id'] != '-':
        p = f"{' '.join(row['product'].split()[:5])}...{' '.join(row['product'].split()[-5:])}"
    else:
        p = row['product']
    return p
act['product'] = act.apply(fix_long_prod_names, axis = 1)
cart['product'] = cart.apply(fix_long_prod_names, axis = 1)
ans['product'] = ans.apply(fix_long_prod_names, axis = 1)

def fix_long_exp_names(row):
    if len(row['experiment_name'].split()) > 3:
        return f"{' '.join(row['experiment_name'].split()[:3])}..."
    else:
        return row['experiment_name']
    
act['experiment_name'] = act.apply(fix_long_exp_names, axis = 1)
cart['experiment_name'] = cart.apply(fix_long_exp_names, axis = 1)
ans['experiment_name'] = ans.apply(fix_long_exp_names, axis = 1)
vis['experiment_name'] = vis.apply(fix_long_exp_names, axis = 1)

multiquestions = ans[(ans.type.isin(['multiple', 'single'])) & (ans.position != 'screener')].question.unique().tolist()
open_questions = ans[(ans.type.isin(['open', 'open_end'])) & (ans.position != 'screener')].question.unique().tolist()
scaled_questions = ans[(ans.type == 'scaled') & (ans.position != 'screener')].question.unique().tolist()
screener_questions = ans[ans.position == 'screener'].question.unique().tolist()

def make_flat(l2d):
    new_list = list(itertools.chain.from_iterable(l2d))
    return new_list
screener_options = {}
for q in screener_questions:
    if ans[(ans.answer != '-') & (ans.question == q)].answer.nunique() > 1:
        if ans[ans.question == q].type.unique()[0] == 'multiple':
            screener_options[q] = ans[(ans.answer != '-') & (ans.question == q)].answer.unique().tolist()
    #         all_replies = ans[(ans.answer != '-') & (ans.question == q)].answer.unique().tolist()
    #         screener_options[q] = [*set(make_flat([i.strip() for i in all_replies.split('|')]))]
        if ans[ans.question == q].type.unique()[0] == 'single':
            screener_options[q] = ans[(ans.answer != '-') & (ans.question == q)].answer.unique().tolist()
    
if len(open_questions) == 0:
    open_questions = ['-']
    
if len(scaled_questions) == 0:
    open_questions = ['-']
    
image_path = './data/banner_12mb.jpg'
# Using base64 encoding and decoding
def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
        return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

load_figure_template("SUPERHERO")

app = Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO, dbc_css])#

heading = html.Div([
    html.Img(src=b64_image(image_path), style={'width': '100%', 'height': 'auto'}),#
    html.H1("Project name: 21 video analysis", className="bg-primary")
    ],
)

# https://dashcheatsheet.pythonanywhere.com/

breakouts_filter = dbc.Card(
            [
                dbc.Row([dbc.Col(html.H5('Select breakouts (optionally)'))], className='dbc'),
                dbc.Row(
                [
                    *[dbc.Col(
                        [html.Label([q.capitalize()], style={'font-weight': 'bold', "text-align": "center"}),
                         dcc.Dropdown(
                        sorted(screener_options[q]) + ['All options'],
                        ['All options'],
                        multi=True,
                        optionHeight=45,#default = 35
                        id=f"{q}")],
                        width = 2,
                        className='dbc'
                            ) for q in list(screener_options.keys())[:6]],
                    *[dbc.Col(
                        [html.Label([q.capitalize()], style={'font-weight': 'bold', "text-align": "center"}),
                         dcc.Dropdown(
                        sorted(screener_options[q]) + ['All options'],
                        ['All options'],
                        multi=True,
                        optionHeight=45,#default = 35
                        id=f"{q}")],
                        width = 2,
                        className='dbc'
                            ) for q in list(screener_options.keys())[6:]],

                        ]
                        )],
            className="transparent-card",
            style={'background-color': 'transparent', 
               'box-shadow': 'none', 
               'border': '1px solid #f28718',
                'margin-top': 5,
                'padding-top': 5,
              },
        )
performance_row = dbc.Container([
    # filter col 1
    dbc.Col(dbc.Card([
        html.Div([html.H5('At what level would you like to filter the data?')]),
        html.Div([
            dcc.RadioItems(
                        options = ['brand', 'product'], value = 'brand',
                        id="radio_grouped_parameter_perf",
                        className="dbc")
        ], className='dbc'),
        html.Div([html.H5('BRAND / PRODUCT')], className='dbc'),
        html.Div([
            dcc.Dropdown(
                        id="dd_perf",
                        value = ['All options'],
                        multi=True,
                        optionHeight=45,
                        className='dbc')
        ], className = 'dbc'),
        
        
    ], 
            className="h-100 transparent-card",
            style={'background-color': 'transparent', 
               'box-shadow': 'none', 
               'border': '1px solid #f28718',
                'margin-top': 5,
                'padding-top': 5,
              }
    
    ), width = 3, className='d-flex flex-column h-100'),
     #end of first column (filters)
    # column 2: line graph
    dbc.Col([
        dbc.Card([
            #rowq one: LINE CHART
            dbc.CardBody(dcc.Graph(id = 'line_opening_buying', className='h-100'), 
                         className='transparent-card dbc')
            ], 
            style = {'height':'100%', 'background-color': 'transparent'}, 
            className = 'transparent-card')
    ], className='d-flex flex-column h-100', width = 5),
    # column 3bar charts 
    dbc.Col([
        dbc.Card([
            # row1: share bars
            dbc.CardBody(dcc.Graph(id = 'bar_shares', className='h-100'), className='transparent-card dbc'),
            ],
            className='d-flex flex-column transparent-card',
            style = {'height':'50%', 'background-color': 'transparent'}),
        dbc.Card([
            # row2: per 100 bars
            dbc.CardBody(dcc.Graph(id = 'bar_values_per_hundred', className='h-100'), className='transparent-card dbc'),
            ], 
            className='d-flex flex-column transparent-card',
            style = {'height':'50%', 'background-color': 'transparent'})
        
    ], width = 4, className='d-flex flex-column h-100'),
], 
    className='d-flex', 
    style={'height': '100vh', 'margin-left':0, 'margin-right':0, 'padding-left':0, 'padding-right':0}, 
    fluid=True
)



donnat_bubble_row = dbc.Container([
    # filter col 1
    dbc.Col(dbc.Card([
        html.Div([html.H5('At what level would you like to filter the data?')]),
        html.Div([
            dcc.RadioItems(
                        options = ['brand', 'product'], value = 'brand',
                        id="radio_grouped_parameter_bubble",
                        className="dbc")
        ], className='dbc'),
        html.Div([html.H5('BRAND / PRODUCT')], className='dbc'),
        html.Div([
            dcc.Dropdown(
                        id="dd_bubble",
                        value = ['All options'],
                        multi=True,
                        optionHeight=45,
                        className='dbc')
        ], className = 'dbc'),
        
        html.Div([html.H5('Select experiment(s)')], className='dbc'),
        html.Div([
            dcc.Dropdown(
                        act['experiment_name'].unique().tolist() + ['All experiments'],
                        ['All experiments'],
                        multi=True,
                        optionHeight=45,#default = 35
                        id="dd_exp_bubble",
                        )
        ], className='dbc'),
        
    ], 
            className="h-100 transparent-card",
            style={'background-color': 'transparent', 
               'box-shadow': 'none', 
               'border': '1px solid #f28718',
                'margin-top': 5,
                'padding-top': 5,
              }
    
    ), width = 3, className='d-flex flex-column h-100'),
     #end of first column (filters)
    # column 2: line graph
    dbc.Col([
        dbc.Card([
            #rowq one: LINE CHART
            dbc.CardBody(dcc.Graph(id = 'bubble', className='h-100'), className='transparent-card dbc')
            ], 
            style = {'height':'100%', 'background-color': 'transparent'}, 
            className = 'transparent-card')
    ], className='d-flex flex-column h-100', width = 5),
    # column 3: bar charts 
    dbc.Col([
        dbc.Card([
            # row1: share bars
            dbc.CardBody(dcc.Graph(id = 'donnat_quantity', className='h-100'), className='transparent-card dbc'),
            ],
            className='d-flex flex-column transparent-card',
            style = {'height':'50%', 'background-color': 'transparent'}),
        dbc.Card([
            # row2: per 100 bars
            dbc.CardBody(dcc.Graph(id = 'donnat_amount', className='h-100'), className='transparent-card dbc'),
            ], 
            className='d-flex flex-column transparent-card',
            style = {'height':'50%', 'background-color': 'transparent'})
        
    ], width = 4, className='d-flex flex-column h-100'), 
], 
    className='d-flex', 
    style={'height': '100vh', 'margin-left':0, 'margin-right':0, 'padding-left':0, 'padding-right':0}, 
    fluid=True
)
    
ps_row = dbc.Container([
    # filter col 1
    dbc.Col(dbc.Card([
        html.Div([html.H5('At what level would you like to filter the data?')]),
        html.Div([
            dcc.RadioItems(
                        options = ['brand', 'product'], value = 'brand',
                        id="radio_grouped_parameter_ps",
                        className="dbc")
        ], className='dbc'),
        html.Div([html.H5('BRAND / PRODUCT')], className='dbc'),
        html.Div([
            dcc.Dropdown(
                        id="dd_ps",
                        value = ['All options'],
                        multi=True,
                        optionHeight=45,
                        className='dbc')
        ], className = 'dbc'),
        
        html.Div([html.H5('Select experiment(s)')], className='dbc'),
        html.Div([
            dcc.Dropdown(
                        act['experiment_name'].unique().tolist() + ['All experiments'],
                        ['All experiments'],
                        multi=True,
                        optionHeight=45,#default = 35
                        id="dd_exp_ps",
                        )
        ], className='dbc'),
        
        
    ], 
            className="h-100 transparent-card",
            style={'background-color': 'transparent', 
               'box-shadow': 'none', 
               'border': '1px solid #f28718',
                'margin-top': 5,
                'padding-top': 5,
              }
    
    ), width = 3, className='d-flex flex-column h-100'),
     #end of first column (filters)
    # column 2: plot
    dbc.Col([
        dbc.Card([
            html.H5('Post-Shopping (Multiple choice questions)', className='card-title dbc',
                    style={'textAlign': 'center'}),
            # row1: bar chart
            dbc.CardBody([dcc.Dropdown(
                        multiquestions,
                        multiquestions[0],
                        multi=False,
                        optionHeight=120,#default = 35
                        id="dd_question_ps",
                        )], className='transparent-card dbc'),
            dbc.CardBody(dcc.Graph(id = 'ps_bar', className='h-100'), className='transparent-card dbc'),
            ],
            className='d-flex flex-column transparent-card',
            style = {'height':'100%', 'background-color': 'transparent'}),
        
    ], width = 9, className='d-flex flex-column h-100'), 
], 
    className='d-flex', 
    style={'height': '100vh', 'margin-left':0, 'margin-right':0, 'padding-left':0, 'padding-right':0}, 
    fluid=True
)

# scaled_row = dbc.Container([
#     # filter col 1
#     dbc.Col(dbc.Card([
#         html.Div([html.H5('At what level would you like to filter the data?')]),
#         html.Div([
#             dcc.RadioItems(
#                         options = ['brand', 'product'], value = 'brand',
#                         id="radio_grouped_parameter_scaled",
#                         className="dbc")
#         ], className='dbc'),
#         html.Div([html.H5('BRAND / PRODUCT')], className='dbc'),
#         html.Div([
#             dcc.Dropdown(
#                         id="dd_scaled",
#                         value = ['All options'],
#                         multi=True,
#                         optionHeight=45,
#                         className='dbc')
#         ], className = 'dbc'),
        
#         html.Div([html.H5('Select experiment(s)')], className='dbc'),
#         html.Div([
#             dcc.Dropdown(
#                         act['experiment_name'].unique().tolist() + ['All experiments'],
#                         ['All experiments'],
#                         multi=True,
#                         optionHeight=45,#default = 35
#                         id="dd_exp_scaled",
#                         )
#         ], className='dbc'),
        
        
#     ], 
#             className="h-100 transparent-card",
#             style={'background-color': 'transparent', 
#                'box-shadow': 'none', 
#                'border': '1px solid #f28718',
#                 'margin-top': 5,
#                 'padding-top': 5,
#               }
    
#     ), width = 3, className='d-flex flex-column h-100'),
#      #end of first column (filters)
#     dbc.Col([
#         dbc.Card([
#             html.H5('Post-Shopping (Scaled questions)', className='card-title dbc',
#                     style={'textAlign': 'center'}),
#             # row1: bar chart
#             dbc.CardBody([dcc.Dropdown(
#                         scaled_questions,
#                         scaled_questions[0],
#                         multi=False,
#                         optionHeight=120,#default = 35
#                         id="dd_question_scaled",
#                         )], className='transparent-card dbc'),
#             dbc.CardBody(dcc.Graph(id = 'scaled_bar', className='h-100'), className='transparent-card dbc'),
#             ],
#             className='d-flex flex-column transparent-card',
#             style = {'height':'100%', 'background-color': 'transparent'}),
        
#     ], width = 9, className='d-flex flex-column h-100'), 
# ], 
#     className='d-flex', 
#     style={'height': '100vh', 'margin-left':0, 'margin-right':0, 'padding-left':0, 'padding-right':0}, 
#     fluid=True
# )

wc_row = html.Div(dbc.Row([
        dbc.Col([
                    html.P('Wordcloud (Open answer questions)', style={'textAlign': 'left'}),
                    html.Hr(),
                    dbc.Col(dcc.Dropdown(
                        open_questions,
                        open_questions[0],
                        multi=False,
                        optionHeight=120,#default = 35
                        id="dd_open_wc",
                        )),
                    dbc.Row([html.H5('Select experiment(s)')], className='dbc'),
                    dbc.Row([
                        dcc.Dropdown(
                                    act['experiment_name'].unique().tolist() + ['All experiments'],
                                    ['All experiments'],
                                    multi=True,
                                    optionHeight=45,#default = 35
                                    id="dd_exp_wc",
                                    )], className = 'dbc'),
                    html.P('Add stopwords (optionally).\nPlease, use no punctuation marks.', style={'textAlign': 'left'}),
                    html.Hr(),
                    dcc.Textarea(
                        id='add_stopwords',
                        value='',
                        style={'width': '100%', 'height': 20},
                    ),
                    html.P('Remove stopwords (optionally, also to return previously removed words).\nPlease, use no punctuation marks.', style={'textAlign': 'left'}),
                    html.Hr(),
                    dcc.Textarea(
                        id='remove_stopwords',
                        value='',
                        style={'width': '100%', 'height': 20},
                    ),
                    html.Button('Submit', id='submit_button', n_clicks=0, className = 'dbc'),
                ], md=3, className = 'dbc'),
        #conditional
        dbc.Col(id = 'conditional_output', md=9, className = 'dbc')
                          ]), 
                  style={
        'padding-right': 5,
        'padding-left': 5,
        'margin-right': 5,
        'margin-left': 5,
    })



app.layout = dbc.Container(
                           fluid=True,#use all the width 
                           children=[
                                heading,
                                breakouts_filter,
                                performance_row,
                                donnat_bubble_row,
                                ps_row,
#                                 scaled_row,
                                wc_row
                           ], 
                           style={'padding-right': 5, 'padding-left': 5}
)

# Function to map categories to colors
def get_color_map(categories, colors):
    color_map = {}
    unique_categories = sorted(set(categories))
    for i, category in enumerate(unique_categories):
        color_map[category] = colors[i % len(colors)]
    return color_map

color_map = {
    'product': get_color_map(act[act['product'] != '-']['product'].unique().tolist(), 
                             sns.color_palette("Spectral", act[act['product'] != '-']['product'].nunique()+1).as_hex()),
    'brand': get_color_map(act[act.brand != '-']['brand'].unique().tolist(), 
                           sns.color_palette("Spectral", act[act.brand != '-']['brand'].nunique()+1).as_hex()),
    
    'experiment_name': get_color_map(act['experiment_name'].unique().tolist(), 
                           sns.color_palette("Spectral", act['experiment_name'].nunique()+1).as_hex()),
}
# color_map['product'].update({"<br>".join(wrap(k, 25)):v for k,v in color_map['product'].items()})

@app.callback(Output('dd_perf', 'options'), Input('radio_grouped_parameter_perf', 'value'))
def dd_brand_product_perf(radio):
    if radio == 'brand':
        output = ['All options'] + act[act['brand'] != '-']['brand'].unique().tolist()
    else:
        output = ['All options'] + act[act['product'] != '-']['product'].unique().tolist()
    return output

@app.callback(Output('dd_bubble', 'options'), Input('radio_grouped_parameter_bubble', 'value'))
def dd_brand_product_bubble(radio):
    if radio == 'brand':
        output = ['All options'] + act[act['brand'] != '-']['brand'].unique().tolist()
    else:
        output = ['All options'] + act[act['product'] != '-']['product'].unique().tolist()
    return output

@app.callback(Output('dd_ps', 'options'), Input('radio_grouped_parameter_ps', 'value'))
def dd_brand_product_ps(radio):
    if radio == 'brand':
        output = ['All options'] + act[act['brand'] != '-']['brand'].unique().tolist()
    else:
        output = ['All options'] + act[act['product'] != '-']['product'].unique().tolist()
    return output

# @app.callback(Output('dd_scaled', 'options'), Input('radio_grouped_parameter_scaled', 'value'))
# def dd_brand_product_scaled(radio):
#     if radio == 'brand':
#         output = ['All options'] + act[act['brand'] != '-']['brand'].unique().tolist()
#     else:
#         output = ['All options'] + act[act['product'] != '-']['product'].unique().tolist()
#     return output

# OPENING/BUYING RATE line
@app.callback(
    # Set the input and output of the callback to link the dropdown to the graph
    Output(component_id='line_opening_buying', component_property='figure'),
    Output(component_id='bar_shares', component_property='figure'),
    Output(component_id='bar_values_per_hundred', component_property='figure'),
    [Input(component_id='dd_perf', component_property='value'),
    Input(component_id='radio_grouped_parameter_perf', component_property='value')] +\
    [Input(f"{q}", "value") for q in list(screener_options.keys())]
)
def plot_by_experiment(input_dd, grouped_parameter, *selected_options):
    if grouped_parameter == 'brand':
        if 'All options' in input_dd or len(input_dd) == 0:
            df_actions = act.copy(deep=True)
        else:
            df_actions = act[(act['brand'].isin(input_dd)) | (act['brand'] == '-')]
            
        radio_filter = df_actions[df_actions.brand != '-'].brand.unique().tolist()
    else:
        if 'All options' in input_dd or len(input_dd) == 0:
            df_actions = act.copy(deep=True)
        else:
            df_actions = act[(act['product'].isin(input_dd)) | (act['product'] == '-')]
        radio_filter = df_actions[df_actions['product'] != '-']['product'].unique().tolist()

        
    # BREAKOUTS FILTERING
    # Convert input tuple to list of selected options per question
    selected_options_per_question = list(selected_options)
    # Get the list of questions to filter on
    questions_to_filter = list(screener_options.keys())
    # Start with all user_ids
    valid_user_ids = set(ans['uid'].unique())

    for question, selected_answers in zip(questions_to_filter, selected_options_per_question):
        if 'All options' not in selected_answers or len(selected_answers) == 0:
            # Filter responses to get user_ids for current question based on selected answers
            filtered_user_ids = set(ans[(ans['question'] == question) 
                                        & (ans['answer'].isin(selected_answers))]['uid'].unique())
            
            # Intersection with valid_user_ids to progressively narrow down the user_ids based on each question's selected answers
            valid_user_ids = valid_user_ids.intersection(filtered_user_ids)

    # Now filter the logs DataFrame to keep only rows with user_ids in valid_user_ids
    df_actions = df_actions[df_actions['uid'].isin(valid_user_ids)]

    # CONTINUE WITH FILTERED DATA
    base = df_actions.groupby('experiment_name').uid.nunique().reset_index()
    # openers of any product
    df_openers = df_actions[(df_actions.action == 'view') 
                            & (df_actions.page_type == 'product')]
    #buyeres of any product
    all_buyers = df_actions[(df_actions.action == 'reached') & (df_actions.page_type == 'checkout')].uid.unique()
    # any openers buyers
    openers_buyers = df_actions[df_actions['is_opener_buyer'] == 'opened_bought']['uid'].unique()
    # in case when people should open product page to buy and All brands/SKUs selected, opening rate will be 100%. 
    # if buying is obligatory and All brands/SKUs selected, buying rate will be 100%. 
    
    nb_pr_openers_per_leg = (
        df_openers[df_openers[grouped_parameter].isin(radio_filter)]
        .groupby('experiment_name')
        .uid.nunique().sort_values(ascending = False).reset_index()
    )
    
    # total buyers
    nb_pr_buyers_per_leg = (
        cart[(cart.uid.isin(all_buyers)) 
             & (cart[grouped_parameter].isin(radio_filter))]
        .groupby('experiment_name')
        .uid.nunique().sort_values(ascending = False).reset_index()
    )
    
    #openers buyers
    nb_pr_openers_buyers_per_leg = (
        cart[(cart.uid.isin(openers_buyers)) 
             & (cart[grouped_parameter].isin(radio_filter))]
        .groupby('experiment_name')
        .uid.nunique().sort_values(ascending = False).reset_index()
    )
    
    quantity_total = (
        cart[(cart.uid.isin(all_buyers))]
        .groupby('experiment_name')
        .quantity.sum().reset_index()
    )
    quantity_products = (
        cart[(cart.uid.isin(all_buyers)) & (cart[grouped_parameter].isin(radio_filter))]
        .groupby('experiment_name')
        .quantity.sum().reset_index()
    )
    amount_total = (
        cart[(cart.uid.isin(all_buyers))]
        .groupby('experiment_name')
        .amount.sum().reset_index()
    )
    amount_products = (
        cart[(cart.uid.isin(all_buyers)) & (cart[grouped_parameter].isin(radio_filter))]
        .groupby('experiment_name')
        .amount.sum().reset_index()
    )
    
    amount_set = {}
    quantity_set = {}
    for exp in cart.experiment_name.unique():
        amount_set[exp] = (
        cart[(cart.uid.isin(all_buyers)) 
             & (cart['experiment_name'] == exp)
             & (cart[grouped_parameter].isin(radio_filter))]
        .groupby('uid')
        .amount.sum()
        .values.tolist()
        )
        quantity_set[exp] = (
        cart[(cart.uid.isin(all_buyers)) 
             & (cart['experiment_name'] == exp)
             & (cart[grouped_parameter].isin(radio_filter))]
        .groupby('uid')
        .quantity.sum()
        .values.tolist()
        )
     
    
    merged_data = (
        pd.merge(nb_pr_openers_per_leg, 
                 nb_pr_buyers_per_leg, 
                 on='experiment_name', 
                 suffixes=('_product_openers', '_product_buyers'))
    ).rename(columns = {'uid_product_openers':'product_openers', 'uid_product_buyers':'product_buyers'})
    merged_data = merged_data.merge(nb_pr_openers_buyers_per_leg, 
                                    on='experiment_name').rename(columns = {'uid':'product_openers_buyers'})
    merged_data = merged_data.merge(base, how = 'left',
                                    on='experiment_name').rename(columns = {'uid':'base'})
    
    for col, metric, df in zip(['quantity_total', 'quantity_products', 'amount_total', 'amount_products'],
                             ['quantity', 'quantity', 'amount', 'amount'],
                            [quantity_total, quantity_products, amount_total, amount_products]):
        merged_data = merged_data.merge(df, how = 'left',
                                    on='experiment_name').rename(columns = {metric:col})
        
    
    merged_data['Opening rate'] = np.round(merged_data['product_openers']/merged_data['base']*100, 2)
    merged_data['Buying rate'] = np.round(merged_data['product_buyers']/merged_data['base']*100, 2)
    merged_data['Conversion'] = np.round(merged_data['product_openers_buyers']/merged_data['product_openers']*100, 2)
    merged_data['Share of choices'] = np.round(merged_data['quantity_products']/merged_data['quantity_total']*100, 2)
    merged_data['Share of value'] = np.round(merged_data['amount_products']/merged_data['amount_total']*100, 2)
    
    merged_data['mean_quantity'] = merged_data['experiment_name'].apply(lambda x: np.mean(quantity_set[x]))
    merged_data['mean_amount'] = merged_data['experiment_name'].apply(lambda x: np.mean(amount_set[x]))
    
    merged_data['Value per 100 respondents'] = merged_data['Buying rate'] * merged_data['mean_amount']
    merged_data['Volume per 100 respondents'] = merged_data['Buying rate'] * merged_data['mean_quantity']
    
    merged_data['experiment_name'] = merged_data['experiment_name'].apply(lambda t: "<br>".join(wrap(t, 25)))
    
    ## FIGURE 1: OPENING BUYING CONVERSION
    fig = go.Figure()

    # Add the new ratio line with a secondary y-axis
    fig.add_trace(
        go.Bar(
            x=merged_data['experiment_name'],
            y=merged_data['Opening rate'],
            name='Opening rate',
            yaxis='y',
            text=[f"Experiment: {exp}<br>Opening Rate: {rate}%<br>Total uids: {uids}<br>Nb openers: {nb}<br>Conversion: {conv}%" 
              for exp, rate, uids, nb, conv in zip(merged_data['experiment_name'], merged_data['Opening rate'], 
                                             merged_data['base'], merged_data['product_openers'], merged_data['Conversion'])],
            hoverinfo='text',  # Use 'text' for hover information
            textposition='none',
        )
    )
    fig.add_trace(
        go.Bar(
            x=merged_data['experiment_name'],
            y=merged_data['Buying rate'],
            name='Buying rate',
            yaxis='y',
            text=[f"Experiment: {exp}<br>Buying Rate: {rate}%<br>Total uids: {uids}<br>Nb buyers: {nb}<br>Conversion: {conv}%" 
              for exp, rate, uids, nb, conv in zip(merged_data['experiment_name'], merged_data['Buying rate'], 
                                         merged_data['base'],merged_data['product_buyers'], merged_data['Conversion'])],
            hoverinfo='text',  # Use 'text' for hover information
            textposition='none',
        )
    )
#     # Add the new ratio line with a secondary y-axis
#     fig.add_trace(
#         go.Bar(
#             x=merged_data['experiment_name'],
#             y=merged_data['Conversion'],
#             name='Conversion',
#             yaxis='y2',
#             opacity = 0.4,
#             width = 0.1,
#             text=[f"Experiment: {exp}<br>Conversion: {rate}%<br>Total uids: {uids}" 
#               for exp, rate, uids in zip(merged_data['experiment_name'], merged_data['Conversion'], merged_data['base'])],
#             hoverinfo='text',  # Use 'text' for hover information
#             textposition='none',
#         )
#     )

    fig.update_layout(
        title='Opening & Buying Rates',
        legend=dict(
            orientation="h",
             x = 1, 
            y = 120,
             yanchor="top",
             xanchor="center",
        ),
#         yaxis2=dict(
#             title='Conversion, %',
#             overlaying='y',
#             side='right',
# #             titlefont=dict(color='rgba(219, 64, 82, 0.7)'), 
#             anchor='x'
#         ),
        xaxis=dict(
            title='Experiment Name',
#             showgrid=False,
        ),
#         yaxis=dict(
#             showgrid=False,
#         ),
        margin=dict(t=100, b=0, l=0, r=0),
        plot_bgcolor='rgba(255,255,255,0)',  # Light background for the plot area
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the entire figure
        
    )
    
    
    # # FIGURE 2: SHARE OF CHOICES SHARE OF VALUE
    fig2 = go.Figure()
    
    # Add the first bar trace for 'Share of choices'
    fig2.add_trace(
        go.Bar(
            x=merged_data['experiment_name'],
            y=merged_data['Share of choices'],
            name='Share of choices',
#             marker=dict(color='rgba(50, 171, 96, 0.7)'),
            offsetgroup=1,
#             width=0.2,
            text=[f"Experiment: {exp}<br>Share of choices: {rate}%<br>Quantity total: {t}<br>Quantity products: {nb}" 
              for exp, rate, t, nb in zip(merged_data['experiment_name'], merged_data['Share of choices'], 
                                         merged_data['quantity_total'],merged_data['quantity_products'])],
            hoverinfo='text',  # Use 'text' for hover information
            textposition='none',
        )
    )

    # Add the second bar trace for 'Share of value'
    # Note: This will automatically create a secondary y-axis on the right
    fig2.add_trace(
        go.Bar(
            x=merged_data['experiment_name'],
            y=merged_data['Share of value'],
            name='Share of value',
#             marker=dict(color='rgba(219, 64, 82, 0.7)'),  # Example color, set your own
            offsetgroup=2,
#             width=0.2,
            yaxis='y',
            text=[f"Experiment: {exp}<br>Share of value: {rate}%<br>Amount total: {t}<br>Amount products: {np.round(nb, 2)}" 
              for exp, rate, t, nb in zip(merged_data['experiment_name'], merged_data['Share of value'], 
                                         merged_data['amount_total'],merged_data['amount_products'])],
            hoverinfo='text',  # Use 'text' for hover information
            textposition='none',
        )
    )
    # Update the layout to adjust the appearance and the axes
    fig2.update_layout(
        title='Share of Choices & Share of Value',
        barmode='group',  # This ensures that bars are grouped next to each other
        bargap=0.1,  # Space between bars within a group
        bargroupgap=0.05,  # Space between groups
        legend=dict(
            orientation="h",
             x = 1, y = 120,
             yanchor="top",
             xanchor="center",
        ),
        yaxis=dict(
            title='Share, %',
#             titlefont=dict(color='rgba(50, 171, 96, 0.7)'), 
        ),
        xaxis=dict(
            title='Experiment Name'
        ),
        margin=dict(t=100, b=0, l=0, r=0)
    )
    
    ## FIGURE 3: VOLUME/VALUE PER 100 RESPONDENTS
    fig3 = go.Figure()
    
    # Add the first bar trace for 'Share of choices'
    fig3.add_trace(
        go.Bar(
            x=merged_data['experiment_name'],
            y=merged_data['Value per 100 respondents'],
            name='Value per 100 respondents',
#             marker=dict(color='rgba(50, 171, 96, 0.7)'),  # Example color, set your own
            offsetgroup=1,
            text=[f"Experiment: {exp}<br>Value per 100 respondents: {np.round(rate, 2)}" 
              for exp, rate in zip(merged_data['experiment_name'], merged_data['Value per 100 respondents'])], 
            hoverinfo='text',  # Use 'text' for hover information
            textposition='none',
        )
    )

    # Add the second bar trace for 'Share of value'
    # Note: This will automatically create a secondary y-axis on the right
    fig3.add_trace(
        go.Bar(
            x=merged_data['experiment_name'],
            y=merged_data['Volume per 100 respondents'],
            name='Volume per 100 respondents',
#             marker=dict(color='rgba(219, 64, 82, 0.7)'),  # Example color, set your own
            offsetgroup=2,
            yaxis='y2',
            text=[f"Experiment: {exp}<br>Volume per 100 respondents: {np.round(rate, 2)}" 
              for exp, rate in zip(merged_data['experiment_name'], merged_data['Volume per 100 respondents'])], 
            hoverinfo='text',  # Use 'text' for hover information
            textposition='none',
        )
    )
    # Update the layout to adjust the appearance and the axes
    fig3.update_layout(
        title='Volume & Value generated per 100 respondents',
        barmode='group',  # This ensures that bars are grouped next to each other
        bargap=0.1,  # Space between bars within a group
        bargroupgap=0.05,  # Space between groups
        legend=dict(
            orientation="h",
             x = 1, y = 110,
             yanchor="top",
             xanchor="center",
        ),
        yaxis=dict(
#             title='Value per 100 respondents',
#             titlefont=dict(color='rgba(50, 171, 96, 0.7)'), 
        ),
        yaxis2=dict(
#             title='Volume per 100 respondents',
            overlaying='y',
            side='right',
#             titlefont=dict(color='rgba(219, 64, 82, 0.7)'), 
            anchor='x'
        ),
        xaxis=dict(
            title='Experiment Name'
        ),
        margin=dict(t=120, b=0, l=0, r=0)
    )                
                    
    return fig, fig2, fig3


# #second plot line
@app.callback(
    # Set the input and output of the callback to link the dropdown to the graph
    Output(component_id='donnat_quantity', component_property='figure'),
    Output(component_id='donnat_amount', component_property='figure'),
    [Input(component_id='dd_exp_bubble', component_property='value'),
    Input(component_id='radio_grouped_parameter_bubble', component_property='value')] +\
    [Input(f"{q}", "value") for q in list(screener_options.keys())]
)
def plots_line2(input_exp, grouped_parameter, *selected_options):
    if  'All experiments' in input_exp or len(input_exp) == 0:
        df_actions = act.copy(deep=True)
    else:
        df_actions = act[(act['experiment_name'].isin(input_exp))]

    # BREAKOUTS FILTERING
    # Convert input tuple to list of selected options per question
    selected_options_per_question = list(selected_options)
    # Get the list of questions to filter on
    questions_to_filter = list(screener_options.keys())
    # Start with all user_ids
    valid_user_ids = set(df_actions['uid'].unique())

    for question, selected_answers in zip(questions_to_filter, selected_options_per_question):
        if 'All options' not in selected_answers or len(selected_answers) == 0:
            # Filter responses to get user_ids for current question based on selected answers
            filtered_user_ids = set(ans[(ans['question'] == question) 
                                        & (ans['answer'].isin(selected_answers))]['uid'].unique())
            
            # Intersection with valid_user_ids to progressively narrow down the user_ids based on each question's selected answers
            valid_user_ids = valid_user_ids.intersection(filtered_user_ids)

    # Now filter the logs DataFrame to keep only rows with user_ids in valid_user_ids
    df_actions = df_actions[df_actions['uid'].isin(valid_user_ids)]

    
    ## FIGURE 1-2: donnats
    all_buyers = df_actions[(df_actions.action == 'reached') & (df_actions.page_type == 'checkout')].uid.unique()
    qty_per_product = (
        cart[cart.uid.isin(all_buyers)]
        .groupby(grouped_parameter)
        .quantity.sum().sort_values(ascending = False).reset_index()
    )
    total = qty_per_product['quantity'].sum()
    qty_per_product['quantity'] = qty_per_product['quantity'].apply(lambda x: np.round(x/total * 100, 2))
    
    amount_per_product = (
        cart[cart.uid.isin(all_buyers)]
        .groupby(grouped_parameter)
        .amount.sum().sort_values(ascending = False).reset_index()
    )
    total = amount_per_product['amount'].sum()
    amount_per_product['amount'] = amount_per_product['amount'].apply(lambda x: np.round(x/total * 100, 2))
    
    fig1 = (
        px.pie(
            data_frame=qty_per_product, 
            names=grouped_parameter, 
            values='quantity',
            hole = 0.8,
            color = grouped_parameter,
            color_discrete_map=color_map[grouped_parameter],
            category_orders={f"{grouped_parameter}": qty_per_product[grouped_parameter].values.tolist()},
#             width=200, height=200,
        )
        .update_traces(textposition='none', 
                       textinfo='none',
                       hovertemplate = "Product/brand:%{label}<br>Quantity bought:%{value}, %<extra></extra>"
                      )
        .update_layout(showlegend=False, hoverlabel_align = 'right',
                       margin=dict(t=10, l=10, r=10, b=10),
                       autosize=True,
                      # Add annotations in the center of the donut pies.
                        annotations=[dict(text=f'Share of choices<br>{"<br>".join(input_exp)}', 
                                      x=0.5, y=0.5, font_size=16, showarrow=False)],
                      )
    )
    
    fig2 = (
        px.pie(
            data_frame=amount_per_product, names=grouped_parameter, 
            values='amount',
            hole = 0.8,
            color = grouped_parameter,
            color_discrete_map=color_map[grouped_parameter],
            category_orders={f"{grouped_parameter}": amount_per_product[grouped_parameter].values.tolist()},
#             width=200, height=200,
                  )
        .update_traces(textposition='none', 
                       textinfo='none',
                       hovertemplate = "Product/brand:%{label}<br>Amount spent:%{value}, %<extra></extra>"
                      )
        .update_layout(showlegend=False, hoverlabel_align = 'right',
                       margin=dict(b=10, l=10, r=10, t=10),
                       autosize=True,
                      # Add annotations in the center of the donut pies.
                        annotations=[dict(text=f'Share of value<br>{"<br>".join(input_exp)}', 
                                      x=0.5, y=0.5, font_size=16, showarrow=False)],
#                        title={
#                             'text': 'Share of value',
#                             'y':0.9,
#                             'x':0.5,
#                             'xanchor': 'center',
#                             'yanchor': 'top'}
                      )
            )
    
    ## FIGURE 4: wordcloud
    fig4 = go.Figure()             
                    
    return fig1, fig2

# PS MULTI answers BAR
@app.callback(
    # Set the input and output of the callback to link the dropdown to the graph
    Output(component_id='ps_bar', component_property='figure'),
#     Output('table1', 'children'),
    [Input(component_id='dd_exp_ps', component_property='value'),
     Input(component_id='radio_grouped_parameter_ps', component_property='value'),
     Input(component_id='dd_question_ps', component_property='value'),
    Input(component_id='dd_ps', component_property='value'),
    ] +\
    [Input(f"{q}", "value") for q in list(screener_options.keys())]
    
)
def bar_ps_plot(input_exp, grouped_parameter, input_question, input_dd, *selected_options):
    if 'All experiments' in input_exp or len(input_exp) == 0:
        df_answers = ans.copy(deep=True)
        exp_filter = 'All experiments'
    else:
        df_answers = ans[ans['experiment_name'].isin(input_exp)]
        exp_filter = '<br>'.join(sorted([' '.join(i.split()[:4]) for i in input_exp]))
    
    df_answers = df_answers[df_answers.question == input_question]
    all_options = [i for i in ans[ans.question == input_question].answer.unique() if i != '-']
    
    if grouped_parameter == 'brand' and 'All options' not in input_dd:
        df_answers = df_answers[df_answers.brand.isin(input_dd)]
    if grouped_parameter == 'product' and 'All options' not in input_dd:
        df_answers = df_answers[df_answers['product'].isin(input_dd)]
        
    # BREAKOUTS FILTERING
    # Convert input tuple to list of selected options per question
    selected_options_per_question = list(selected_options)
    # Get the list of questions to filter on
    questions_to_filter = list(screener_options.keys())
    # Start with all user_ids
    valid_user_ids = set(df_answers['uid'].unique())

    for q, selected_answers in zip(questions_to_filter, selected_options_per_question):
        if 'All options' not in selected_answers or len(selected_answers) == 0:
            # Filter responses to get user_ids for current question based on selected answers
            filtered_user_ids = set(ans[(ans['question'] == q) 
                                        & (ans['answer'].isin(selected_answers))]['uid'].unique())
            
            # Intersection with valid_user_ids to progressively narrow down the user_ids based on each question's selected answers
            valid_user_ids = valid_user_ids.intersection(filtered_user_ids)

    # Now filter the logs DataFrame to keep only rows with user_ids in valid_user_ids
    df_answers = df_answers[df_answers['uid'].isin(valid_user_ids)]
    
    base = (
        df_answers[['uid', 'question', 'product_id', 'experiment_name']]
        .drop_duplicates()
        .groupby('experiment_name').uid.nunique()
        .reset_index()
    )
    
    grouped_df = (
        df_answers
        .groupby(['answer', 'experiment_name']).uid.count()
        .reset_index().sort_values(by = 'uid', ascending = False)
    )
    
    grouped_df = grouped_df.merge(base, how = 'left', on = 'experiment_name', suffixes = ('','_base'))
    grouped_df['percent'] = grouped_df['uid']/grouped_df['uid_base']*100
    
    for opt in all_options:
        for exp in df_answers.experiment_name.unique():
            if grouped_df.query('answer == @opt & experiment_name == @exp').shape[0] == 0:
                df_opt = pd.DataFrame({'answer':[opt], 'experiment_name':[exp], 'uid':[0], 'uid_base':[0], 'percent':[0]})
                grouped_df = pd.concat([grouped_df, df_opt]).reset_index(drop=True)
                
    sort_list = grouped_df.groupby('answer', as_index = False).percent.sum().sort_values(by = ['percent'], 
                                                            ascending = True).answer.values.tolist()
    grouped_df['answer'] = pd.Categorical(grouped_df['answer'], categories=sort_list, ordered=True)
    # Sort the DataFrame by the 'column_to_sort'
    grouped_df = grouped_df.sort_values('answer')
    # Now, the tickvals and ticktext will be in the correct order
    tickvals_and_text = grouped_df.answer.values.tolist()
    
    bar_plot = (
        px.bar(
            grouped_df,
            y='answer',
            x='percent',
            color = 'experiment_name',
            barmode="group",
            orientation='h',
            labels={'percent':'Repliers, %', 'answer':'Answer'},
            title=f"{input_question}",
        )
        .update_layout(showlegend=False, 
                       hoverlabel_align = 'left',
                       margin=dict(t=30, b=0, l=0, r=0),
                       yaxis=dict(
                            tickmode='array',  # Set tick mode to 'array'
                            tickvals=tickvals_and_text,
                            ticktext=tickvals_and_text
                        ),
                      )
    )
    
    
#     data = grouped_df.to_dict('rows')
#     columns =  [{"name": i, "id": i,} for i in (grouped_df.columns)]
#     t = dash_table.DataTable(data=data, columns=columns)
    
    return bar_plot

# # PS SCALED answers BAR
# @app.callback(
#     # Set the input and output of the callback to link the dropdown to the graph
#     Output(component_id='scaled_bar', component_property='figure'),
#     [Input(component_id='dd_exp_scaled', component_property='value'),
#      Input(component_id='radio_grouped_parameter_scaled', component_property='value'),
#      Input(component_id='dd_question_scaled', component_property='value'),
#     Input(component_id='dd_scaled', component_property='value'),
#     ] +\
#     [Input(f"{q}", "value") for q in list(screener_options.keys())]
    
# )
# def bar_scaled_plot(input_exp, grouped_parameter, input_question, input_dd, *selected_options):
#     if 'All experiments' in input_exp or len(input_exp) == 0:
#         df_answers = ans.copy(deep=True)
#         exp_filter = 'All experiments'
#     else:
#         df_answers = ans[ans['experiment_name'].isin(input_exp)]
#         exp_filter = '<br>'.join(sorted([' '.join(i.split()[:4]) for i in input_exp]))
    
#     df_answers = df_answers[df_answers.question == input_question]
#     all_options = [i for i in ans[ans.question == input_question].answer.unique() if i != '-']
    
#     if grouped_parameter == 'brand' and 'All options' not in input_dd:
#         df_answers = df_answers[df_answers.brand.isin(input_dd)]
#     if grouped_parameter == 'product' and 'All options' not in input_dd:
#         df_answers = df_answers[df_answers['product'].isin(input_dd)]
        
#     # BREAKOUTS FILTERING
#     # Convert input tuple to list of selected options per question
#     selected_options_per_question = list(selected_options)
#     # Get the list of questions to filter on
#     questions_to_filter = list(screener_options.keys())
#     # Start with all user_ids
#     valid_user_ids = set(df_answers['uid'].unique())

#     for q, selected_answers in zip(questions_to_filter, selected_options_per_question):
#         if 'All options' not in selected_answers or len(selected_answers) == 0:
#             # Filter responses to get user_ids for current question based on selected answers
#             filtered_user_ids = set(ans[(ans['question'] == q) 
#                                         & (ans['answer'].isin(selected_answers))]['uid'].unique())
            
#             # Intersection with valid_user_ids to progressively narrow down the user_ids based on each question's selected answers
#             valid_user_ids = valid_user_ids.intersection(filtered_user_ids)

#     # Now filter the logs DataFrame to keep only rows with user_ids in valid_user_ids
#     df_answers = df_answers[df_answers['uid'].isin(valid_user_ids)][
#         ['uid', 'question', 'product_id', 'experiment_name']]
    
#     base = (
#         df_answers
#         .drop_duplicates(subset = ['uid', 'product_id'])
#         .groupby('experiment_name').uid.count()
#         .reset_index()
#     )
    
#     grouped_df = (
#         df_answers
#         .groupby(['answer', 'experiment_name']).uid.count()
#         .reset_index().sort_values(by = 'uid', ascending = False)
#     )
    
#     grouped_df = grouped_df.merge(base, how = 'left', on = 'experiment_name', suffixes = ('','_base'))
#     grouped_df['percent'] = grouped_df['uid']/grouped_df['uid_base']*100
    
#     for opt in all_options:
#         for exp in df_answers.experiment_name.unique():
#             if grouped_df.query('answer == @opt & experiment_name == @exp').shape[0] == 0:
#                 df_opt = pd.DataFrame({'answer':[opt], 'experiment_name':[exp], 'uid':[0], 'uid_base':[0], 'percent':[0]})
#                 grouped_df = pd.concat([grouped_df, df_opt]).reset_index(drop=True)
                
#     sort_list = grouped_df.groupby('answer', as_index = False).percent.sum().sort_values(by = ['percent'], 
#                                                             ascending = True).answer.values.tolist()
#     grouped_df['answer'] = pd.Categorical(grouped_df['answer'], categories=sort_list, ordered=True)
#     # Sort the DataFrame by the 'column_to_sort'
#     grouped_df = grouped_df.sort_values('answer')
#     # Now, the tickvals and ticktext will be in the correct order
#     tickvals_and_text = grouped_df.answer.values.tolist()
    
#     bar_plot = (
#         px.bar(
#             grouped_df,
#             y='answer',
#             x='percent',
#             color = 'experiment_name',
#             barmode="group",
#             orientation='h',
#             labels={'percent':'Repliers, %', 'answer':'Answer'},
#             title=f"{input_question}",
#         )
#         .update_layout(showlegend=False, 
#                        hoverlabel_align = 'left',
#                        margin=dict(t=30, b=0, l=0, r=0),
#                        yaxis=dict(
#                             tickmode='array',  # Set tick mode to 'array'
#                             tickvals=tickvals_and_text,
#                             ticktext=tickvals_and_text
#                         ),
#                       )
#     )
    
#     return bar_plot


# BUBLE
@app.callback(
    # Set the input and output of the callback to link the dropdown to the graph
    Output(component_id='bubble', component_property='figure'),
#     Output('table1', 'children'),
    [Input(component_id='dd_exp_bubble', component_property='value'),
     Input(component_id='radio_grouped_parameter_bubble', component_property='value'),
    Input(component_id='dd_bubble', component_property='value'),
    ] +\
    [Input(f"{q}", "value") for q in list(screener_options.keys())]
    
)
def bubble_plot(input_exp, grouped_parameter, input_dd, *selected_options):
    if 'All experiments' in input_exp or len(input_exp) == 0:
        df_actions = act.copy(deep=True)
        df_cart = cart.copy(deep=True)
    else:
        df_actions = act[act['experiment_name'].isin(input_exp)]
        df_cart = cart[cart['experiment_name'].isin(input_exp)]
    
    product_id_list = df_actions.product_id.unique().tolist()
    
    if grouped_parameter == 'brand' and 'All options' not in input_dd:
        product_id_list = df_actions[df_actions.brand.isin(input_dd)].product_id.unique().tolist()
    if grouped_parameter == 'product' and 'All options' not in input_dd:
        product_id_list = df_actions[df_actions['product'].isin(input_dd)].product_id.unique().tolist()
      
        
    # BREAKOUTS FILTERING
    # Convert input tuple to list of selected options per question
    selected_options_per_question = list(selected_options)
    # Get the list of questions to filter on
    questions_to_filter = list(screener_options.keys())
    # Start with all user_ids
    valid_user_ids = set(df_actions['uid'].unique())

    for q, selected_answers in zip(questions_to_filter, selected_options_per_question):
        if 'All options' not in selected_answers or len(selected_answers) == 0:
            # Filter responses to get user_ids for current question based on selected answers
            filtered_user_ids = set(ans[(ans['question'] == q) 
                                        & (ans['answer'].isin(selected_answers))]['uid'].unique())
            
            # Intersection with valid_user_ids to progressively narrow down the user_ids based on each question's selected answers
            valid_user_ids = valid_user_ids.intersection(filtered_user_ids)

    # Now filter the logs DataFrame to keep only rows with user_ids in valid_user_ids
    df_actions = df_actions[df_actions['uid'].isin(valid_user_ids)]
    df_cart =  df_cart[df_cart['uid'].isin(valid_user_ids)]   
        
    all_buyers = df_actions[(df_actions.action == 'reached') & (df_actions.page_type == 'checkout')].uid.unique()
    cart_buying = df_cart[(df_cart.uid.isin(all_buyers))
                         & (df_cart.product_id.isin(product_id_list))][['quantity', 'amount', 'price', 'brand', 'product']]

    fig = (
        px.scatter(
            cart_buying, 
            y = 'quantity',
            x = 'amount',
#             color = 'product',
            size = 'price',
            title = 'Quantity & Amount bought by Price',
            labels={'quantity': 'Quantity', 'amount': 'Amount', 'price':'Price'} # Capitalize axis labels
        )
        .update_layout(
            showlegend=True, 
            hoverlabel_align='left',
            xaxis_title='Amount',  # Capitalize X-axis title
            yaxis_title='Quantity',  # Capitalize Y-axis title
            # Uncomment and set values for margin if needed
            margin=dict(t=30, b=0, l=0, r=0),
                      )
            )

    return fig


# wordcloud chart
@app.callback(
    # Set the input and output of the callback to link the dropdown to the graph
    Output(component_id='conditional_output', component_property='children'),#figure
    [Input(component_id='dd_exp_wc', component_property='value'),
     Input(component_id='dd_open_wc', component_property='value'),
        Input('submit_button', 'n_clicks'),
        State('add_stopwords', 'value'),
        State('remove_stopwords', 'value'),
    ] +\
    [Input(f"{q}", "value") for q in list(screener_options.keys())]
    
)
def wordcloud(input_exp, input_question, n_clicks, added, removed, *selected_options):
    if 'All experiments' in input_exp or len(input_exp) == 0:
        df_answers = ans.copy(deep=True)
        exp_filter = 'All experiments'
    else:
        df_answers = ans[ans['experiment_name'].isin(input_exp)]
        exp_filter = '<br>'.join(sorted([' '.join(i.split()[:4]) for i in input_exp]))
    
    df_answers = df_answers[df_answers.question == input_question]
    
    # BREAKOUTS FILTERING
    # Convert input tuple to list of selected options per question
    selected_options_per_question = list(selected_options)
    # Get the list of questions to filter on
    questions_to_filter = list(screener_options.keys())
    # Start with all user_ids
    valid_user_ids = set(df_answers['uid'].unique())

    for q, selected_answers in zip(questions_to_filter, selected_options_per_question):
        if 'All options' not in selected_answers or len(selected_answers) == 0:
            # Filter responses to get user_ids for current question based on selected answers
            filtered_user_ids = set(ans[(ans['question'] == q) 
                                        & (ans['answer'].isin(selected_answers))]['uid'].unique())
            
            # Intersection with valid_user_ids to progressively narrow down the user_ids based on each question's selected answers
            valid_user_ids = valid_user_ids.intersection(filtered_user_ids)

    # Now filter the logs DataFrame to keep only rows with user_ids in valid_user_ids
    df_answers = df_answers[df_answers['uid'].isin(valid_user_ids)]
    stop_words = set(pd.read_csv('english_stopwords.csv')['word'].unique())
    
    if n_clicks > 0:
        added_set = set([i.strip() for i in added.split()])
        removed_set = set([i.strip() for i in removed.split()])
        stop_words.update(added_set)
        stop_words = stop_words.difference(removed_set)
    
    if len(df_answers) > 0:
        wc = calc_wordcloud(df_answers, stop_words, translation=None, width=2000, height=1000, 
#                             font_path="assets/Arial Unicode.ttf"
                            bg_color = 'white', color_pal = 'Spectral')

        # Convert the word cloud to a numpy array
        wordcloud_image = wc.to_array()

        # Convert numpy array to PIL Image
        wordcloud_image = Image.fromarray(wordcloud_image)

        buffer = io.BytesIO()
        wc.to_image().save(
            buffer, format="png"
        )
        buffer.seek(0)
        
        
        
        output = html.Div([
#             html.P(used_stopwords),
            dbc.Card(html.Img(
            src = "data:image/png;base64,{}".format(base64.b64encode(buffer.read()).decode())
        ), style = {'margin':30}),
        ] )
        
    else:
        output = html.P('No data available for building the wordcloud!')
        
    return output


if __name__ == '__main__':
    app.run_server(debug=True)