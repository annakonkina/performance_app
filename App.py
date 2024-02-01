from dash import Dash, dash_table
from dash import dcc, html
from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate

from dash_bootstrap_templates import load_figure_template
import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import pandas as pd
import numpy as np
from textwrap import wrap
import matplotlib.pyplot as plt

import os
from threading import Timer
import webbrowser

from wordcloud import WordCloud
from deep_translator import GoogleTranslator
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from PIL import Image
import base64


from share_functions import *
from word_cloud import *


dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

vis = pd.read_csv("./vis_sample.csv")
act = pd.read_csv("./act_sample.csv")
ans = pd.read_csv("./ans_sample.csv")
cart = pd.read_csv("./cart_sample.csv")

multiquestions = ans[ans.type == 'multiple'].question.unique().tolist()
open_questions = ans[ans.type == 'open'].question.unique().tolist()
if len(open_questions) == 0:
#     open_questions = ['-']
    open_questions = ['Q1. Could you please select all the reasons that made you buy this product?']
    
image_path = './banner_12mb.jpg'
# Using base64 encoding and decoding
def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
        return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

load_figure_template("SUPERHERO")

app = Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO, dbc_css])#, 'assets/custom.css'

heading = html.Div([
    html.Img(src=b64_image(image_path), style={'width': '100%', 'height': 'auto'}),#
    html.H1("Project name: 21 video analysis", className="bg-primary text-white p-2")
    ])

# heading = html.H1("Project name: 21 video analysis", className="bg-primary text-white p-2")

# https://dashcheatsheet.pythonanywhere.com/
dropdown = dbc.Container(
    [
        dbc.Row(
        [
            dbc.Col(html.P('By which experiments you would like to filter the data?'),width = 6),
            dbc.Col(html.P('At which level you want to show the data?'),width = 6),
        ]
        ),
        dbc.Row(
        [
            dbc.Col(dcc.Dropdown(
            vis.experiment_name.unique().tolist() + ['All experiments'],
            ['All experiments'],
            multi=True,
            id="experiment_dd",
            className="dbc"),width = 6),
            
            dbc.Col(dcc.RadioItems(
            options = ['brand', 'product'], value = 'brand',
            id="radio_grouped_parameter",
            className="dbc"),width = 6),
        ]
        )
    ]
)

graphs = html.Div(
    [
        #first row of graphs: 6 shares
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id = 'line_opening_buying'), className="p-1", width=4),
                
                dbc.Col(dcc.Graph(id = 'donnat_qty',
                                 style={'width': '100%', 'height': '70%'}), className="p-1", width=2),
                dbc.Col(dcc.Graph(id = 'donnat_amount',
                                 style={'width': '100%', 'height': '70%'}), className="p-1", width=2),
                
                dbc.Col([
                    dcc.RadioItems(
                            options = ['value', 'volume'], value = 'value',
                            id="radio_val_vol",
                            className="p-1"),
                    dbc.Col(dcc.Graph(id = 'bar_value_volume'), className="p-1"),
                ], width=4),
            ], className='dbc'
        ),
        #2d row of graphs
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id = 'bubble_chart'), width=6),
                dbc.Col([
                    dcc.Dropdown(multiquestions, multiquestions[0], 
                                 id='dropdown_question',className="p-1"),
                    dbc.Col(dcc.Graph(id = 'bar_ps'), className="p-1"),
                ], width=6),
            ],
            className='dbc',
        ),
        #3d row of graphs
        dbc.Row(
            [
                dbc.Col(dcc.Markdown('**Note:** The size of the bubbles in the plot represents the price.',
                                    style={'fontSize': 10}), width=6),
#                 dbc.Col(dbc.Card("Width equal if not specified"), width=6),
            ],
            className='dbc',
        ),
        
        # 4th row of graphs
        dbc.Row(
            [
                dbc.Col([
                    dcc.Dropdown(open_questions, open_questions[0], 
                                 id='dropdown_open_questions',className="p-1"),
                    dbc.Col(dcc.Graph(id = 'wordcloud_plot'), className="p-1"),
                ], width=12),
            ],
            className='dbc',
        ),
        
    ], className = 'dbc'
)

app.layout = dbc.Container(fluid=True, children=[heading,dropdown, graphs])#

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
                           sns.color_palette("Spectral", act[act.brand != '-']['brand'].nunique()+1).as_hex())
}
color_map['product'].update({"<br>".join(wrap(k, 25)):v for k,v in color_map['product'].items()})
color_map['brand'].update({"<br>".join(wrap(k, 25)):v for k,v in color_map['brand'].items()}) 

# https://plotly.com/python/setting-graph-size/
# SHARE OF CHOICES/VALUE SHARE
@app.callback(
    # Set the input and output of the callback to link the dropdown to the graph
    Output(component_id='donnat_qty', component_property='figure'),
    Output(component_id='donnat_amount', component_property='figure'),
    Input(component_id='experiment_dd', component_property='value'),
    Input(component_id='radio_grouped_parameter', component_property='value')
)
def donnat_qty_plot(input_exp, grouped_parameter):
    if 'All experiments' in input_exp or len(input_exp) == 0:
        df_actions = act.copy(deep=True)
        exp_filter = 'All experiments'
    else:
        df_actions = act[act['experiment_name'].isin(input_exp)]
        exp_filter = '<br>'.join(sorted([' '.join(i.split()[:4]) for i in input_exp]))
        
    all_buyers = df_actions[(df_actions.action == 'reached') & (df_actions.page_type == 'checkout')].uid.unique()
    qty_per_product = (
        cart[cart.uid.isin(all_buyers)]
        .groupby([grouped_parameter])
        .quantity.sum().sort_values(ascending = False).reset_index()
    )
    total = qty_per_product['quantity'].sum()
    qty_per_product['quantity'] = qty_per_product['quantity'].apply(lambda x: np.round(x/total * 100, 2))
    qty_per_product[grouped_parameter] = qty_per_product[grouped_parameter].apply(lambda t: "<br>".join(wrap(t, 25)))
    
    amount_per_product = (
        cart[cart.uid.isin(all_buyers)]
        .groupby(grouped_parameter)
        .amount.sum().sort_values(ascending = False).reset_index()
    )
    total = amount_per_product['amount'].sum()
    amount_per_product['amount'] = amount_per_product['amount'].apply(lambda x: np.round(x/total * 100, 2))
    amount_per_product[grouped_parameter] = amount_per_product[grouped_parameter].apply(lambda t: "<br>".join(wrap(t, 25)))
    
    
    donat_qty = (
        px.pie(
            data_frame=qty_per_product, names=grouped_parameter, 
            values='quantity',
            hole = 0.8,
            color = grouped_parameter,
            color_discrete_map=color_map[grouped_parameter],
            category_orders={f"{grouped_parameter}": qty_per_product[grouped_parameter].values.tolist()},
#             width=200, height=200,
                  )
        .update_traces(textposition='none', 
                       textinfo='none',
#                        hovertemplate = "Product/brand:%{label}: <br>Quantity bought:%{value}, %<extra></extra>"
                      )
        .update_layout(showlegend=False, hoverlabel_align = 'left',margin=dict(t=0, b=0, l=0, r=0),
                       autosize=True,
                      # Add annotations in the center of the donut pies.
                    annotations=[dict(text=f'{grouped_parameter.capitalize()}<br>Value share<br>{exp_filter}.', x=0.5, y=0.5, font_size=12, showarrow=False)]
                      )
            )
    
    donat_amount = (
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
#                        hovertemplate = "Product/brand:%{label}: <br>Amount spent:%{value}, %<extra></extra>"
                      )
        .update_layout(showlegend=False, hoverlabel_align = 'left',margin=dict(t=0, b=0, l=0, r=0),
                       autosize=True,
                      # Add annotations in the center of the donut pies.
                    annotations=[dict(text=f'{grouped_parameter.capitalize()}<br>Share of choices<br> {exp_filter}.', x=0.5, y=0.5, font_size=12, showarrow=False)]
                      )
            )
        
    return donat_qty, donat_amount
  
# OPENING/BUYING RATE line
@app.callback(
    # Set the input and output of the callback to link the dropdown to the graph
    Output(component_id='line_opening_buying', component_property='figure'),
    Input(component_id='experiment_dd', component_property='value'),
    Input(component_id='radio_grouped_parameter', component_property='value')
)
def line_plot_conversion(input_exp, grouped_parameter):
    if 'All experiments' in input_exp or len(input_exp) == 0:
        df_actions = act.copy(deep=True)
        exp_filter = 'All experiments'
    else:
        df_actions = act[act['experiment_name'].isin(input_exp)]
        exp_filter = '<br>'.join(sorted([' '.join(i.split()[:4]) for i in input_exp]))

    base = df_actions.uid.nunique()
    
    df_openers = df_actions[(df_actions.action == 'view') & (df_actions.page_type == 'product')]
    all_buyers = df_actions[(df_actions.action == 'reached') & (df_actions.page_type == 'checkout')].uid.unique()
    openers_buyers = df_actions[df_actions['is_opener_buyer'] == 'opened_bought']['uid'].unique()
    
    nb_openers_per_product = (
        df_openers
        .groupby(grouped_parameter)
        .uid.nunique().sort_values(ascending = False).reset_index()
    )
    nb_openers_per_product[grouped_parameter] = nb_openers_per_product[grouped_parameter].apply(lambda t: "<br>".join(wrap(t, 25)))
    # total buyers
    nb_buyers_per_product = (
        cart[cart.uid.isin(all_buyers)]
        .groupby(grouped_parameter)
        .uid.nunique().sort_values(ascending = False).reset_index()
    )
    nb_buyers_per_product[grouped_parameter] = nb_buyers_per_product[grouped_parameter].apply(lambda t: "<br>".join(wrap(t, 25)))
    #openers buyers
    nb_openers_buyers_per_product = (
        cart[cart.uid.isin(openers_buyers)]
        .groupby(grouped_parameter)
        .uid.nunique().sort_values(ascending = False).reset_index()
    )
    nb_openers_buyers_per_product[grouped_parameter] = (
        nb_openers_buyers_per_product[grouped_parameter].apply(lambda t: "<br>".join(wrap(t, 25)))
    )
    
    merged_data = (
        pd.merge(nb_openers_per_product, 
                 nb_buyers_per_product, 
                 on=grouped_parameter, 
                 suffixes=('_openers', '_buyers_total'))
    )
    merged_data = merged_data.merge(nb_openers_buyers_per_product, on='brand').rename(columns = {'uid':'uid_buyers'})
    
    merged_data['Opening rate'] = merged_data['uid_openers']/base*100
    merged_data['Buying rate'] = merged_data['uid_buyers_total']/base*100
    merged_data['Conversion'] = merged_data['uid_buyers']/merged_data['uid_openers']*100
    
    line_plot = (
        px.line(
            merged_data,
            x=grouped_parameter,
            y=['Opening rate', 'Buying rate'],
            labels={'value':'Rate, %', 'variable':'Rate Type'},
            title=f'Opening and Buying Rate per {grouped_parameter.capitalize()}'
        )
    )
    
    # Create a new figure with secondary y-axis
    fig = go.Figure()

    # Add the existing lines
    for trace in line_plot.data:
        fig.add_trace(go.Scatter(x=trace['x'], y=trace['y'], name=trace['name']))

    # Add the new ratio line with a secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=merged_data[grouped_parameter],
            y=merged_data['Conversion'],
            name='Conversion',
            yaxis='y2'
        )
    )
    # Update the layout with the secondary axis
    fig.update_layout(
        yaxis2=dict(
            title='Conversion',
            overlaying='y',
            side='right',
            showgrid=False,  # Hides the gridlines for the secondary y-axis

        )
    )
    
    fig.update_layout(
                       showlegend=False, 
                       hoverlabel_align = 'left',
                       margin=dict(t=30, b=0, l=0, r=0),
                       yaxis_title='%',
                      )
    
    return fig

# VOLUME PER 100, VALUE PER 100
@app.callback(
    # Set the input and output of the callback to link the dropdown to the graph
    Output(component_id='bar_value_volume', component_property='figure'),
    Input(component_id='experiment_dd', component_property='value'),
    Input(component_id='radio_grouped_parameter', component_property='value'),
    Input(component_id='radio_val_vol', component_property='value'),
)
def bar_value_volume_plot(input_exp, grouped_parameter, metric):
    if 'All experiments' in input_exp or len(input_exp) == 0:
        df_actions = act.copy(deep=True)
        exp_filter = 'All experiments'
    else:
        df_actions = act[act['experiment_name'].isin(input_exp)]
        exp_filter = '<br>'.join(sorted([' '.join(i.split()[:4]) for i in input_exp]))
    
    all_buyers = df_actions[(df_actions.action == 'reached') & (df_actions.page_type == 'checkout')].uid.unique()
    cart_buying = cart[cart.uid.isin(all_buyers)]
    base = df_actions.uid.nunique()
    metric_dict = {
        'value':'amount',
        'volume':'quantity',
    }
    
    grouped_df = generate_grouped_df_metric_100(cart_buying=cart_buying, 
                                   grouped_parameter=grouped_parameter, 
                                   metric = metric_dict[metric], 
                                   base = base, 
                                   label=metric).reset_index()

    grouped_df[grouped_parameter] = grouped_df[grouped_parameter].apply(lambda t: "<br>".join(wrap(t, 25)))
    
    bar_plot = (
        px.bar(
            grouped_df,
            x=grouped_parameter,
            y=metric,
#             labels={'value':'Rate, %', 'variable':'Rate Type'},
            title=f'{metric.capitalize()} generated per 100 respondents per {grouped_parameter.capitalize()}'
        )
        .update_layout(showlegend=False, 
                       hoverlabel_align = 'left',
                       margin=dict(t=30, b=0, l=0, r=0),
                       xaxis_title=grouped_parameter.capitalize(),  
                        yaxis_title=metric.capitalize(),  
                      )
    )
    
    return bar_plot


# BUBLE
@app.callback(
    # Set the input and output of the callback to link the dropdown to the graph
    Output(component_id='bubble_chart', component_property='figure'),
    Input(component_id='experiment_dd', component_property='value'),
)
def bubble_plot(input_exp):
    if 'All experiments' in input_exp or len(input_exp) == 0:
        df_actions = act.copy(deep=True)
        df_cart = cart.copy(deep=True)
        exp_filter = 'All experiments'
    else:
        df_actions = act[act['experiment_name'].isin(input_exp)]
        df_cart = cart[cart['experiment_name'].isin(input_exp)]
        exp_filter = '<br>'.join(sorted([' '.join(i.split()[:4]) for i in input_exp]))
        
        
    all_buyers = df_actions[(df_actions.action == 'reached') & (df_actions.page_type == 'checkout')].uid.unique()
    cart_buying = df_cart[df_cart.uid.isin(all_buyers)][['quantity', 'amount', 'price', 'brand', 'product']]

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


# PS MULTI answers BAR
@app.callback(
    # Set the input and output of the callback to link the dropdown to the graph
    Output(component_id='bar_ps', component_property='figure'),
    Input(component_id='experiment_dd', component_property='value'),
    Input(component_id='dropdown_question', component_property='value'),
)
def bar_ps_plot(input_exp, question):
    if 'All experiments' in input_exp or len(input_exp) == 0:
        df_answers = ans.copy(deep=True)
        exp_filter = 'All experiments'
    else:
        df_answers = ans[ans['experiment_name'].isin(input_exp)]
        exp_filter = '<br>'.join(sorted([' '.join(i.split()[:4]) for i in input_exp]))
    
    df_answers = df_answers[df_answers.question == question]
    base = len(df_answers[['uid', 'product_id']].drop_duplicates())
    grouped_df = df_answers.groupby('answer').uid.count().reset_index().sort_values(by = 'uid', ascending = True)
    grouped_df['percent'] = grouped_df['uid']/base*100
    
    bar_plot = (
        px.bar(
            grouped_df,
            y='answer',
            x='percent',
            orientation='h',
            labels={'percent':'Repliers, %', 'answer':'Answer'},
            title=f'{question.capitalize()}'
        )
        .update_layout(showlegend=False, 
                       hoverlabel_align = 'left',
                       margin=dict(t=30, b=0, l=0, r=0),
                       yaxis=dict(
                            tickmode='array',  # Set tick mode to 'array'
                            tickvals=grouped_df.answer,  # Position ticks at these points
                            ticktext=grouped_df['answer']  # Label ticks with these texts
                        )
                      )
    )
    
    return bar_plot

# WORDCLOUD
@app.callback(
    # Set the input and output of the callback to link the dropdown to the graph
    Output(component_id='wordcloud_plot', component_property='figure'),
    Input(component_id='experiment_dd', component_property='value'),
    Input(component_id='dropdown_open_questions', component_property='value'),
)
def wc_plot(input_exp, question):
    if 'All experiments' in input_exp or len(input_exp) == 0:
        df_answers = ans.copy(deep=True)
        exp_filter = 'All experiments'
    else:
        df_answers = ans[ans['experiment_name'].isin(input_exp)]
        exp_filter = '<br>'.join(sorted([' '.join(i.split()[:4]) for i in input_exp]))
    
    df_answers = df_answers[df_answers.question == question]
    stop_words = set(stopwords.words('english'))
    if len(df_answers) > 0:
        wc = calc_wordcloud(df_answers, stop_words, translation=None)

        # Convert the word cloud to a numpy array
        wordcloud_image = wc.to_array()

        # Convert numpy array to PIL Image
        wordcloud_image = Image.fromarray(wordcloud_image)

        # Convert PIL Image to Plotly figure
        fig = px.imshow(wordcloud_image)
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0),
                         xaxis=dict(showticklabels=False, visible=False),
                        yaxis=dict(showticklabels=False, visible=False))
        
    else:
        fig = px.bar(title=f'No open questions available...')
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)