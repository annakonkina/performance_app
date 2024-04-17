from dash import Dash, dash_table, no_update
from dash import dcc, html
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate

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
from statsmodels.stats.proportion import proportions_ztest
import math

import os
from threading import Timer
from textwrap import wrap
import base64
import io
import string

from wordcloud import WordCloud
from deep_translator import GoogleTranslator
import nltk
from nltk.stem import WordNetLemmatizer
from whitenoise import WhiteNoise   #for serving static files on Heroku

from share_functions import *
from word_cloud import *
from stat_tests import stat_tests_proportions
from build_opening_rate_table import *
from build_buying_rate_table import *
from build_share_table import *

# # Cheetsheet: https://dashcheatsheet.pythonanywhere.com/
nltk.download('wordnet')
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
image_path = './data/banner_12mb.jpg'
# Using base64 encoding and decoding
def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
        return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

load_figure_template("superhero")

app = Dash(__name__, 
           external_stylesheets=[dbc.themes.SUPERHERO, dbc_css],
          meta_tags=[
              {"name": "viewport", "content": "width=device-width, initial-scale=1"}
          ],
      )
# Reference the underlying flask app (Used by gunicorn webserver in Heroku production deployment)
server = app.server 
# Enable Whitenoise for serving static files from Heroku (the /static folder is seen as root by Heroku) 
server.wsgi_app = WhiteNoise(server.wsgi_app, root='static/') 

# Example usage
vis = pd.read_csv("./data/vis_dashboard_52.csv", low_memory=False)
act = pd.read_csv("./data/act_dashboard_52.csv", low_memory=False)
ans = pd.read_csv("./data/ans_dashboard_52.csv", low_memory=False)
cart = pd.read_csv("./data/cart_dashboard_52.csv", low_memory=False)

act['timestamp'] = pd.to_datetime(act['timestamp'])
act['product_id'] = act['product_id'].apply(lambda x: int(float(x)) if x != '-' else x)
cart['product_id'] = cart['product_id'].apply(lambda x: int(float(x)) if x != '-' else x)
ans['question'] = ans['question'].apply(lambda x: x.replace('.', ' '))

multiquestions = ans[(ans.type.isin(['multiple', 'single'])) & (ans.position != 'screener')].question.unique().tolist()
open_questions = ans[(ans.type.isin(['open', 'open_end'])) & (ans.position != 'screener')].question.unique().tolist()
screener_questions = ans[ans.position == 'screener'].question.unique().tolist()

def make_flat(l2d):
    new_list = list(itertools.chain.from_iterable(l2d))
    return new_list

def check_prop(p1ch, n1ch, p2ch, n2ch):
    if p1ch == '-':
        p1ch = 0
    if p2ch == '-':
        p2ch = 0
    if n1ch == '-':
        n1ch = 0
    if n2ch == '-':
        n2ch = 0

    stat, p = proportions_ztest(np.array([p1ch, p2ch]), np.array([n1ch, n2ch]),
                                alternative="larger")
    if math.isnan(p):
        return True
    if not math.isnan(p) and p <= 0.05:
        return False # Reject -> there is a difference
    else:
        return True # Accept -> there is no difference

screener_options = {}
screener_options_height = {}
for q in screener_questions:
    if ans[(ans.answer != '-') & (ans.question == q)].answer.nunique() > 1:
        if ans[ans.question == q].type.unique()[0] == 'multiple':
            screener_options[q] = ans[(ans.answer != '-') & (ans.question == q)].answer.unique().tolist()
        if ans[ans.question == q].type.unique()[0] == 'single':
            screener_options[q] = ans[(ans.answer != '-') & (ans.question == q)].answer.unique().tolist()
            
        longest_a = max(ans[(ans.answer != '-') & (ans.question == q)].answer.unique().tolist(), key=len)
        if len(longest_a) <= 25:
            screener_options_height[q] = 45
        elif len(longest_a) > 25 and len(longest_a) <= 50:
            screener_options_height[q] = 90
        else:
            screener_options_height[q] = 140
        
if len(open_questions) == 0:
    open_questions = ['-']

heading = html.Div([
    html.Img(src=b64_image(image_path), style={'width': '100%', 'height': 'auto'}),
    html.H1("Project name: 52", className="bg-primary")
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
                        optionHeight=screener_options_height[q],
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
                        optionHeight=screener_options_height[q],
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
                'padding': '15px',
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
                        value = ['Total category'],
                        multi=True,
                        optionHeight=45,
                        className='dbc')
        ], className = 'dbc'),
        
        
    ], 
            className="h-100 transparent-card",
            style={'background-color': 'transparent', 
               'box-shadow': 'none', 
               'border': '1px solid #0F2537',
                   'padding': '15px',
                   'height':'auto',
              }
    
    ), width = 2, className='d-flex flex-column h-100'),
     #end of first column (filters)
    # column 2: line graph
    dbc.Col([
        dbc.Card([
            #rowq one: LINE CHART
            dbc.CardBody(dcc.Graph(id = 'opening_rate_bar', className='h-100'), className='transparent-card dbc')
            ], 
            style = {'height':'100%', 'background-color': 'transparent'}, 
            className = 'transparent-card'),
        dbc.Card([
            #rowq one: LINE CHART
            dbc.CardBody(dcc.Graph(id = 'buying_rate_bar', className='h-100'), className='transparent-card dbc')
            ], 
            style = {'height':'100%', 'background-color': 'transparent'}, 
            className = 'transparent-card'),
    ], className='d-flex flex-column h-100', width = 5),
    # column 3bar charts 
    dbc.Col([
        dbc.Card([
            # row1: share bars
            dbc.CardBody(dcc.Graph(id = 'bar_share_of_choices', className='h-100'), className='transparent-card dbc'),
            ],
            className='d-flex flex-column transparent-card',
            style = {'height':'50%', 'background-color': 'transparent'}),
        dbc.Card([
            # row2: per 100 bars
            dbc.CardBody(dcc.Graph(id = 'bar_share_of_value', className='h-100'), className='transparent-card dbc'),
            ], 
            className='d-flex flex-column transparent-card',
            style = {'height':'50%', 'background-color': 'transparent'})
        
    ], width = 5, className='d-flex flex-column h-100'),
], 
    className='d-flex', 
    style={'height': 'auto', 
           'margin-left':0, 'margin-right':0, 'padding-left':0, 'padding-right':0
          }, 
    fluid=True
)

performance_table_row = dbc.Container([
        # filter col 1
        dbc.Col(dbc.Card([
            html.Div([html.H5('Metric:')]),
            html.Div([
                dcc.RadioItems(
                            options = ['Opening rate', 'Buying rate', 'Share of choices', 'Share of value'], 
                            value = 'Opening rate',
                            id="radio_metric",
                            className="dbc")
            ], className='dbc'),
            html.Div([html.H5('At what level would you like to filter the data?')]),
            html.Div([
                dcc.RadioItems(
                            options = ['brand', 'product'], value = 'brand',
                            id="radio_table_level",
                            className="dbc")
            ], className='dbc'),
            
        ], 
                className="transparent-card",
                style={'background-color': 'transparent', 
                   'box-shadow': 'none', 
                   'border': '1px solid #0F2537',
                       'height':'auto',
                       'padding': '15px',
                  }
        
        ), width = 2, className='d-flex flex-column', style={'flex-grow': 1}),
        
         #end of first column (filters)
        # column 2: table
        dbc.Col([
                dbc.Card([
                    dbc.CardBody(html.Div(id = 'table'))
                    ], 
                    style = {'height':'100%', 'background-color': 'transparent'}, 
                    className = 'transparent-card'),
                # dbc.Row(dbc.Col([
                #     # download button
                #     html.Button("Download CSV", id="btn_csv"),
                #     dcc.Download(id="download-dataframe-csv"),
                # ], width = 2))
                ], className='d-flex flex-column', width = 10, style={'height': '100%'}), 
], #end of container 
    className='d-flex', 
    style={'height': 'auto', 
           'margin-left':0, 'margin-right':0, 'padding-left':0, 'padding-right':0
          },
    fluid=True
)

ps_wc_row = dbc.Container([
    # filter col: one filter for both ps bars and wordcloud
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
                        value = ['Total category'],
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
               'border': '1px solid #0F2537',
                   'padding': '15px',
                   'height':'auto'
              }
    
    ), width = 2, className='d-flex flex-column h-100', style = {'height':'auto'}),
     #end of first column (filters)
    # column 2: plot bars
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
            dbc.CardBody(html.Div(
                            dcc.Graph(id='ps_bar'),
                                        style={
                                            'overflow-y': 'auto',  # Enable vertical scrolling
                                            'height': '500px'  # Set a fixed height for the scrolling container
                                        }
                            )),
            ],
            className='d-flex flex-column transparent-card',
            style = {'height':'100%', 'background-color': 'transparent'}),

        #wordcloud
        dbc.Card([
            html.H5('Wordcloud (Open answer questions)', className='card-title dbc',
                    style={'textAlign': 'center'}),
            # row1: bar chart
            dbc.CardBody([dcc.Dropdown(
                        open_questions,
                        open_questions[0],
                        multi=False,
                        optionHeight=120,#default = 35
                        id="dd_open_wc",
                        )], className='transparent-card dbc'),
            dbc.CardBody([
                # col one within the wordcloud card: filters
                dbc.Row([
                    dbc.Col([
                            html.Hr(),
                            html.P('Add stopwords (optionally).\nPlease, use no punctuation marks, only spaces.', style={'textAlign': 'left'}),
                            html.Hr(),
                            dcc.Textarea(
                                id='add_stopwords',
                                value='',
                                style={'width': '100%', 'height': 20},
                            ),
                            html.P('Remove stopwords (optionally).\nPlease, use no punctuation marks, only spaces.\nThen press "Submit"', style={'textAlign': 'left'}),
                            html.Hr(),
                            dcc.Textarea(
                                id='remove_stopwords',
                                value='',
                                style={'width': '100%', 'height': 20},
                            ),
                            html.Button('Submit', id='submit_button', n_clicks=0, className = 'dbc'),
                            html.P('To reset stopwords list, please press "Reset"', style={'textAlign': 'left'}),
                            html.Button('Reset', id='reset_button', n_clicks=0, className = 'dbc'),
                        ], width=2, className = 'dbc'),
                    # col2 withing wordcloud card: wordcloud image
                    dbc.Col(id = 'conditional_output', width=10, className = 'dbc')
                ]),
                
            ], className='transparent-card dbc'),
            ],
            className='d-flex flex-column transparent-card',
            style = {'height':'100%', 'background-color': 'transparent'}),
        
    ], width = 10, className='d-flex flex-column h-100'), 
], 
    className='d-flex', 
    style={'height': 'auto', 
           'margin-left':0, 'margin-right':0, 'padding-left':0, 'padding-right':0
          }, 
    fluid=True
)

# Data for the table
info_table = pd.DataFrame(
    {"METRICS": [
                "OPENING RATE", 
                "BUYING RATE", 
                "SHARE OF CHOICES", 
                "VALUE SHARE"], 
     "EXPLANATION": [
                "Users opening product page / base size",
                "Users buying product / base size",
                "Number of units bought for a specific product / number of total units bought in store",
                "Value of units bought for a specific product / total value generated in store"
                    ], 
     "OBJECTIVE": [
                "Understand if the product is visible and attractive in a given competitive context",
                "Understand the buying appeal of a product in a given competitive context",
                "Proxy for volume share",
                "Proxy for value share"
                  ],
    }
)

info_dash_table = dbc.Container([
        # column 2: table
        dbc.Col([
                    html.Div(
                            dash_table.DataTable(
                            columns = [{'name': str(i), 'id': str(i)} for i in info_table.columns],
                            data = info_table.to_dict('records'),
                            # Style the table area (you might want to set a background color)
                            style_table={'backgroundColor': 'white',
                                         'overflowY': 'auto'},
                            # Style the headers of the table
                            style_header={
                            'fontWeight': 'bold',
                            'backgroundColor': 'gray',  # You can change this to any color you like
                            'color': 'black'  # Adjust text color for better readability
                            },
                            
                            # Style the cells (if needed, for consistency)
                            style_cell={
                            'textAlign': 'left',
                            'padding': '10px',  # Adjust padding to your preference
                            'backgroundColor': 'white',
                            'color': 'black',  # Adjust text color for better readability
                            'minWidth': '100px',  # Minimum width for each cell
                            'width': 'auto',  # Width of the cell will grow as needed
                            'maxWidth': '300px',  # Maximum width for each cell
                            'whiteSpace': 'normal',  # Allows text to wrap within the cell
                            'height': 'auto',  # Cell height grows with content
                        }),
                    )
                    ], 
                    style = {'height':'100%', 'background-color': 'transparent'}, 
                    className = 'dbc'),
        
], #end of container 
    className='d-flex', 
    style={'height': 'auto', 
           'margin-left':0, 'margin-right':0, 
           'padding':'20px',
          },
    fluid=True
)

app.layout = dbc.Container(
                           fluid=True,#use all the width 
                           children=[
                                heading,
                                breakouts_filter,
                                performance_row,
                                performance_table_row,
                                ps_wc_row,
                                info_dash_table,
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

@app.callback(Output('dd_perf', 'options'), 
              Input('radio_grouped_parameter_perf', 'value')
             )
def dd_brand_product_perf(radio):
    if radio == 'brand':
        output = ['Total category'] + act[act['brand'] != '-']['brand'].unique().tolist()
    else:
        output = ['Total category'] + act[act['product'] != '-']['product'].unique().tolist()
    return output


@app.callback(Output('dd_ps', 'options'), Input('radio_grouped_parameter_ps', 'value'))
def dd_brand_product_ps(radio):
    if radio == 'brand':
        output = ['Total category'] + act[act['brand'] != '-']['brand'].unique().tolist()
    else:
        output = ['Total category'] + act[act['product'] != '-']['product'].unique().tolist()
    return output


# OPENING/BUYING RATE line
@app.callback(
    # Set the input and output of the callback to link the dropdown to the graph
    Output(component_id='opening_rate_bar', component_property='figure'),
    Output(component_id='buying_rate_bar', component_property='figure'),
    Output(component_id='bar_share_of_choices', component_property='figure'),
    Output(component_id='bar_share_of_value', component_property='figure'),
    [Input(component_id='dd_perf', component_property='value'),
    Input(component_id='radio_grouped_parameter_perf', component_property='value')] +\
    [Input(f"{q}", "value") for q in list(screener_options.keys())]
)
def plot_by_experiment(input_dd, grouped_parameter, *selected_options):
    if grouped_parameter == 'brand':
        if 'Total category' in input_dd or len(input_dd) == 0:
            df_actions = act.copy(deep=True)
        else:
            df_actions = act[(act['brand'].isin(input_dd)) | (act['brand'] == '-')]
            
        radio_filter = df_actions[df_actions.brand != '-'].brand.unique().tolist()
    else:
        if 'Total category' in input_dd or len(input_dd) == 0:
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
    
    merged_data['mean_quantity'] = np.round(merged_data['experiment_name'].apply(lambda x: np.mean(quantity_set[x])), 2)
    merged_data['mean_amount'] = np.round(merged_data['experiment_name'].apply(lambda x: np.mean(amount_set[x])), 2)
    
    # merged_data['Value per 100 respondents'] = merged_data['Buying rate'] * merged_data['mean_amount']
    # merged_data['Volume per 100 respondents'] = merged_data['Buying rate'] * merged_data['mean_quantity']
    
    merged_data['experiment_name'] = merged_data['experiment_name'].apply(lambda t: "<br>".join(wrap(t, 25)))
    
    stat_test_dict = {}
    for k, col, b in zip(['Opening rate_stat', 'Buying rate_stat', 'Share of choices_stat', 'Share of value_stat'],
                          ['product_openers', 'product_buyers', 'quantity_products', 'amount_products'],
                         ['base', 'base', 'quantity_total', 'amount_total']
                        ):
        stat_test_dict[k] = {}

        for exp in merged_data['experiment_name'].unique():
        
            stat_test_dict[k][exp] = {
                'number':merged_data.query('experiment_name == @exp')[col].unique()[0],
                'base':merged_data.query('experiment_name == @exp')[b].unique()[0],
                'stat':[],
            }
    
    stat_letters = dict(zip(merged_data.experiment_name.unique(), 
                    string.ascii_uppercase[:merged_data.experiment_name.nunique()]))
    stat_test_dict = stat_tests_proportions(stat_test_dict, stat_letters, no_stat_cols=[], alpha = 0.05)
    
    for k in stat_test_dict.keys():
        merged_data[k] = merged_data['experiment_name'].apply(lambda x: stat_test_dict[k][x]['stat'])

    
    ## FIGURE 1: OPENING BUYING CONVERSION
    fig = go.Figure()

    # Add the new ratio line with a secondary y-axis
    fig.add_trace(
        go.Bar(
            x=merged_data['experiment_name'],
            y=merged_data['Opening rate'],
            name='Opening rate',
            yaxis='y',
            text = [f"{opr:.2f}%<br>{stt}"  # Rounded rate for display on the bar
              for opr, stt in zip(merged_data['Opening rate'], merged_data['Opening rate_stat'])],
            hovertext=[f"Experiment: {exp}<br>Opening Rate: {rate}%<br>Total uids: {uids}<br>Nb openers: {nb}<br>Conversion: {conv}%" 
              for exp, rate, uids, nb, conv in zip(merged_data['experiment_name'], merged_data['Opening rate'], 
                                             merged_data['base'], merged_data['product_openers'], merged_data['Conversion'])],
            hoverinfo='text',  # Use 'text' for hover information
            textposition='inside',
        )
    )
    
    fig.update_layout(
        title='Opening Rates',
        # legend=dict(
        #     orientation="h",
        #      x = 1, 
        #      y = 120,
        #      yanchor="top",
        #      xanchor="center",
        # ),
        xaxis=dict(
            title='Experiment Name',
            tickvals=merged_data['experiment_name'],  # Set the tick values to the experiment names
            ticktext=[f'{exp} ({stat_letters[exp]})<br>{nb}' for exp, nb in zip(merged_data['experiment_name'],
                                                                               merged_data['base'])]
        ),
        yaxis=dict(
            title='Rate, %',
        ),
        margin=dict(t=100, b=0, l=0, r=0),
        plot_bgcolor='rgba(255,255,255,0)',  # Light background for the plot area
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the entire figure
        
    )
    
    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            x=merged_data['experiment_name'],
            y=merged_data['Buying rate'],
            name='Buying rate',
            yaxis='y',
            text = [f"{opr:.2f}%<br>{stt}"  # Rounded rate for display on the bar
              for opr, stt in zip(merged_data['Buying rate'], merged_data['Buying rate_stat'])],
            hovertext=[f"Experiment: {exp}<br>Buying Rate: {rate}%<br>Total uids: {uids}<br>Nb buyers: {nb}<br>Conversion: {conv}%" 
              for exp, rate, uids, nb, conv in zip(merged_data['experiment_name'], merged_data['Buying rate'], 
                                         merged_data['base'],merged_data['product_buyers'], merged_data['Conversion'])],
            hoverinfo='text',  # Use 'text' for hover information
            textposition='inside',
        )
    )

    fig2.update_layout(
        title='Buying Rates',
        xaxis=dict(
            title='Experiment Name',
            tickvals=merged_data['experiment_name'],  # Set the tick values to the experiment names
            ticktext=[f'{exp} ({stat_letters[exp]})<br>{nb}' for exp, nb in zip(merged_data['experiment_name'],
                                                                               merged_data['base'])]
        ),
        yaxis=dict(
            title='Rate, %',
        ),
        margin=dict(t=100, b=0, l=0, r=0),
        plot_bgcolor='rgba(255,255,255,0)',  # Light background for the plot area
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the entire figure
        
    )
    
    
    # # FIGURE 2: SHARE OF CHOICES SHARE OF VALUE
    fig3 = go.Figure()
    
    # Add the first bar trace for 'Share of choices'
    fig3.add_trace(
        go.Bar(
            x=merged_data['experiment_name'],
            y=merged_data['Share of choices'],
            name='Share of choices',
#             marker=dict(color='rgba(50, 171, 96, 0.7)'),
            offsetgroup=1,
#             width=0.2,
            text = [f"{opr:.2f}%<br>{stt}"  # Rounded rate for display on the bar
              for opr, stt in zip(merged_data['Share of choices'], merged_data['Share of choices_stat'])],
            hovertext=[f"Experiment: {exp}<br>Share of choices: {rate}%<br>Quantity total: {t}<br>Quantity products: {nb}" 
              for exp, rate, t, nb in zip(merged_data['experiment_name'], merged_data['Share of choices'], 
                                         merged_data['quantity_total'],merged_data['quantity_products'])],
            hoverinfo='text',  # Use 'text' for hover information
            textposition='inside',
        )
    )

    fig4 = go.Figure()
    # Add the second bar trace for 'Share of value'
    # Note: This will automatically create a secondary y-axis on the right
    fig4.add_trace(
        go.Bar(
            x=merged_data['experiment_name'],
            y=merged_data['Share of value'],
            name='Share of value',
#             marker=dict(color='rgba(219, 64, 82, 0.7)'),  # Example color, set your own
            offsetgroup=2,
#             width=0.2,
            yaxis='y',
            text = [f"{opr:.2f}%<br>{stt}"  # Rounded rate for display on the bar
              for opr, stt in zip(merged_data['Share of value'], merged_data['Share of value_stat'])],
            hovertext=[f"Experiment: {exp}<br>Share of value: {rate}%<br>Amount total: {t}<br>Amount products: {np.round(nb, 2)}" 
              for exp, rate, t, nb in zip(merged_data['experiment_name'], merged_data['Share of value'], 
                                         merged_data['amount_total'],merged_data['amount_products'])],
            hoverinfo='text',  # Use 'text' for hover information
            textposition='inside',
        )
    )
    # Update the layout to adjust the appearance and the axes
    fig3.update_layout(
        title='Share of Choices',
        barmode='group',  # This ensures that bars are grouped next to each other
        # bargap=0.1,  # Space between bars within a group
        # bargroupgap=0.05,  # Space between groups
        # legend=dict(
        #     orientation="h",
        #      x = 1, y = 120,
        #      yanchor="top",
        #      xanchor="center",
        # ),
        yaxis=dict(
            title='Share, %',
#             titlefont=dict(color='rgba(50, 171, 96, 0.7)'), 
        ),
        xaxis=dict(
            title='Experiment Name',
            tickvals=merged_data['experiment_name'],  # Set the tick values to the experiment names
            ticktext=[f'{exp} ({stat_letters[exp]})<br>{nb:.2f}' for exp, nb in zip(merged_data['experiment_name'], merged_data['amount_total'])]
        ),
        margin=dict(t=100, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    fig4.update_layout(
        title='Share of Value',
        barmode='group',  # This ensures that bars are grouped next to each other
        yaxis=dict(
            title='Share, %',
        ),
        xaxis=dict(
            title='Experiment Name',
            tickvals=merged_data['experiment_name'],  # Set the tick values to the experiment names
            ticktext=[f'{exp} ({stat_letters[exp]})<br>{nb}' for exp, nb in zip(merged_data['experiment_name'], merged_data['quantity_total'])]
        ),
        margin=dict(t=100, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig, fig2, fig3, fig4


# performance table line
@app.callback(
    # Set the input and output of the callback to link the dropdown to the graph
    Output(component_id='table', component_property='children'),
    # Output("download-dataframe-csv", "data"),
    [Input(component_id='radio_metric', component_property='value'),
    Input(component_id='radio_table_level', component_property='value')] +\
    [Input(f"{q}", "value") for q in list(screener_options.keys())]
    # [Input("btn_csv", "n_clicks")],
)
def create_table(input_metric, grouped_parameter, *selected_options):
    df_actions = act.copy(deep=True)
    df_cart = cart.copy(deep=True)
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
    df_cart = df_cart[df_cart['uid'].isin(valid_user_ids)]
    
    # GENERATING TABLE NOW
    if input_metric == 'Opening rate':
        df = build_opening_rate_table(df_actions, grouped_parameter, is_percent=True)
        df = df.reset_index().rename(columns = {'index':'Opening rate'})
        df['Opening rate'] = df['Opening rate'].apply(lambda x: x.strip(', %') if 'Base, %' in x else x)
    elif input_metric == 'Buying rate':
        df = build_buying_rate_table(df_actions, df_cart, grouped_parameter, is_percent=True)
        df = df.reset_index().rename(columns = {'index':'Buying rate'})
        df['Buying rate'] = df['Buying rate'].apply(lambda x: x.strip(', %') if 'Base, %' in x else x)
    elif input_metric == 'Share of choices':
        df = (
            build_share_table(df_actions, df_cart, 'quantity', grouped_parameter, is_percent=True)
            .reset_index()
            .rename(columns = {'index':'Share of choices'})
        )
    elif input_metric == 'Share of value':
        df = (
            build_share_table(df_actions, df_cart, 'amount', grouped_parameter, is_percent=True)
            .reset_index()
            .rename(columns = {'index':'Share of value'})
        )
    else:
        df = pd.DataFrame()
     
    
    output = dash_table.DataTable(
            columns = [{'name': str(i), 'id': str(i)} for i in df.columns],
            data = df.to_dict('records'),
            # Style the table area (you might want to set a background color)
            style_table={'backgroundColor': 'white'},
            
            # Style the headers of the table
            style_header={
            'fontWeight': 'bold',
            'backgroundColor': ' #f0af72',  # You can change this to any color you like
            'color': 'black'  # Adjust text color for better readability
            },
            
            # Style the cells (if needed, for consistency)
            style_cell={
            'textAlign': 'left',
            'padding': '10px',  # Adjust padding to your preference
            'backgroundColor': 'white',
            'color': 'black',  # Adjust text color for better readability
            'width': 'auto',
            }
        
        )
    #, dcc.send_data_frame(df.to_csv, "res_table.csv")
    return output

# PS MULTI answers BAR
@app.callback(
    # Set the input and output of the callback to link the dropdown to the graph
    Output(component_id='ps_bar', component_property='figure'),
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
    
    if grouped_parameter == 'brand' and 'Total category' not in input_dd:
        df_answers = df_answers[df_answers.brand.isin(input_dd)]
    if grouped_parameter == 'product' and 'Total category' not in input_dd:
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
        df_answers[['uid', 'product_id', 'experiment_name']].drop_duplicates()
        .groupby('experiment_name').uid.count()
        .reset_index()
    )
    
    grouped_df = (
        df_answers
        .groupby(['answer', 'experiment_name']).uid.count()
        .reset_index().sort_values(by = 'uid', ascending = False)
    )
    grouped_df = grouped_df.merge(base, how = 'left', on = 'experiment_name', suffixes = ('','_base'))
    grouped_df['percent'] = grouped_df['uid']/grouped_df['uid_base']*100
                
    sort_list = grouped_df.groupby('answer', as_index = False).percent.sum().sort_values(by = ['percent'], 
                                                            ascending = True).answer.values.tolist()
    grouped_df['answer'] = pd.Categorical(grouped_df['answer'], categories=sort_list, ordered=True)
    # Sort the DataFrame by the 'column_to_sort'
    grouped_df = grouped_df.sort_values(['answer', 'experiment_name'])

    exps_dfs = []
    for exp in grouped_df.experiment_name.unique():
        exp_df = grouped_df.query('experiment_name == @exp').set_index('answer')[['uid']].rename(columns = {'uid':exp})
        exps_dfs.append(exp_df)
    exps_dfs = pd.concat(exps_dfs, axis = 1).fillna(0)
    # minimum base size of 50 resp per experiment:
    options_to_keep = exps_dfs[(exps_dfs[exps_dfs.columns] >= 50).all(axis=1)].index.values.tolist()
    grouped_df = grouped_df[grouped_df.answer.isin(options_to_keep)]
    
    # Now, the tickvals and ticktext will be in the correct order
    tickvals_and_text = grouped_df.answer.values.tolist()

    # STAT TESTING
    exp_letter = dict(zip(grouped_df.experiment_name.unique(), string.ascii_uppercase[:grouped_df.experiment_name.nunique()]))
    dfs_with_stat = []
    for opt in grouped_df.answer.unique():
        opt_df = grouped_df.query('answer == @opt').set_index('experiment_name')
        all_legs =  [*set(opt_df.index.values)]
        for t1 in all_legs:
            opt_df.loc[t1, 'col_stat'] = '-'
            tested_legs = [i for i in all_legs if i != t1]
            for t2 in tested_legs:
                #DISCRETE DATA
                p1, n1, p2, n2 = opt_df.loc[t1, 'uid'], opt_df.loc[t1, 'uid_base'], \
                                opt_df.loc[t2, 'uid'], opt_df.loc[t2, 'uid_base']

                if p1 != '-' and p2 != '-' and n1 not in ['-', 0] and n2 not in ['-', 0]:
                    if p1 >= 5 and p2 >= 5:
                        try:
                            if not check_prop(p1, n1, p2, n2): #if not True >> Ha
                                opt_df.loc[t1, 'col_stat'] += exp_letter[t2]
                        except:
                            pass
        dfs_with_stat.append(opt_df)

    grouped_df = pd.concat(dfs_with_stat).reset_index()
    grouped_df['col_stat'] = grouped_df['col_stat'].apply(lambda x: ', '.join(sorted(x.strip('-'))) if x != '-' else x)
    grouped_df['experiment_name'] = grouped_df['experiment_name'].apply(lambda x: f'{x} ({exp_letter[x]})')

    grouped_df['display_text'] = grouped_df.apply(
            lambda row: f"{row['experiment_name']}: {row['percent']:.2f}% {row['col_stat']}", axis=1
        )
    
    bar_plot = (
        px.bar(
            grouped_df,
            y='answer',
            x='percent',
            color = 'experiment_name',
            text='display_text',
            barmode="group",
            orientation='h',
            labels={'percent':'Repliers, %', 'answer':'Answer'},
            title=f"{input_question}",
            custom_data=['uid', 'experiment_name', 'col_stat']
        )
        .update_traces(
            texttemplate='%{text}',  # Format the text display; .2f for 2 decimal places
            textposition='inside',  # Position the text inside the bars
            insidetextfont={'size':14},
            hovertemplate="<br>".join([
                "Answer: %{y}",
                "Experiment: %{customdata[1]}",
                "Percent: %{x:.2f}%",
                "Sign. stat difference with: %{customdata[2]}"
                ])
        )
        .update_layout(showlegend=False, 
                       hoverlabel_align = 'left',
                       margin=dict(t=30, b=0, l=0, r=0),
                       paper_bgcolor='rgba(0,0,0,0)',
                       height = 1200, 
                       yaxis=dict(
                            tickmode='array',  # Set tick mode to 'array'
                            tickvals=tickvals_and_text,
                            ticktext=tickvals_and_text,
                        ),
                      )
    )

    

    return bar_plot

# wordcloud chart
@app.callback(
    # Set the input and output of the callback to link the dropdown to the graph
    Output(component_id='conditional_output', component_property='children'),#figure
    [Input(component_id='dd_exp_ps', component_property='value'),
     Input(component_id='dd_open_wc', component_property='value'),
        Input('submit_button', 'n_clicks'),
         Input('reset_button', 'n_clicks'),
        State('add_stopwords', 'value'),
        State('remove_stopwords', 'value'),
    ] +\
    [Input(f"{q}", "value") for q in list(screener_options.keys())]
    
)
def wordcloud(input_exp, input_question, n_clicks, n_clicks_reset, added, removed, *selected_options):
    if 'All experiments' in input_exp or len(input_exp) == 0:
        df_answers = ans.copy(deep=True)
    else:
        df_answers = ans[ans['experiment_name'].isin(input_exp)]
        
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

    if n_clicks_reset > 0:
        added_set = set()
        removed_set = set()
        stop_words = set(pd.read_csv('english_stopwords.csv')['word'].unique())
        
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
        

# Run flask app
if __name__ == "__main__": 
    app.run_server(debug=False, host='0.0.0.0', port=8050)