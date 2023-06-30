from dash import Dash, dcc, Output, Input, html 
import dash_bootstrap_components as dbc   
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np

from dash_utils import *

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
title = dcc.Markdown(children='# Optimization with large learning rate')

# 1D Toy example

df_1d = pd.read_parquet('dash_data/df_1d.parquet')

lrs_1d = df_1d['lr'].dropna().unique()
lrs_1d.sort()
lrs_1d_marks = {i: lr for i, lr in enumerate(lrs_1d)}

start_points_1d = df_1d['start_point'].dropna().unique()
start_points_1d.sort()
start_points_1d_marks = {i: x for i, x in enumerate(start_points_1d)}

graph_1d = dcc.Graph(
    figure={},
    style={'width': '85vh', 'height': '70vh', 'display': 'inline-block'}
)

graph_1d_ends = dcc.Graph(
    figure=create_fig_1d_ends(df_1d), 
    style={'width': '85vh', 'height': '70vh', 'display': 'inline-block'}
)

lrs_1d_slider = dcc.Slider(
    0,
    len(lrs_1d) - 1,
    step=None,
    marks={k: f'{v:.2E}' for k, v in lrs_1d_marks.items()},
    value=1,
)

start_points_1d_slider = dcc.Slider(
    0,
    len(start_points_1d) - 1,
    step=None,
    marks={k: f'{v:.2f}' for k, v in start_points_1d_marks.items()},
    value=1,
)

@app.callback(
    Output(graph_1d, component_property='figure'),
    Input(lrs_1d_slider, component_property='value'),
    Input(start_points_1d_slider, component_property='value'),
)
def update_1d_graph(lr, start_point):
    return create_fig_1d(lrs_1d_marks[lr], start_points_1d_marks[start_point], df_1d)

# 2D

grid_x = np.load('dash_data/grid_x_2d.npy')
grid_y = np.load('dash_data/grid_y_2d.npy')
grid_z = np.load('dash_data/grid_z_2d.npy')

df_2d_gd = pd.read_parquet('dash_data/df_2d_func_gd.parquet')
lrs_2d = df_2d_gd['lr'].dropna().unique()
lrs_2d.sort()
lrs_2d_marks = {i: lr for i, lr in enumerate(lrs_2d)}

df_2d_ends = pd.read_parquet('dash_data/df_2d_ends.parquet')

graph_2d = dcc.Graph(
    figure={},
    style={'width': '85vh', 'height': '80vh', 'display': 'inline-block'}

)

graph_2d_ends = dcc.Graph(
    figure=create_fig_2d_ends(df_2d_ends),
    style={'width': '85vh', 'height': '80vh', 'display': 'inline-block'}
)

lrs_2d_slider = dcc.Slider(
    0,
    len(lrs_2d) - 1,
    step=None,
    marks={k: f'{v:.2E}' for k, v in lrs_2d_marks.items()},
    value=1,
)

@app.callback(
    Output(graph_2d, component_property='figure'),
    Input(lrs_2d_slider, component_property='value'),
)
def update_2d_graph(lr):
    return create_fig_2d(grid_x, grid_y, grid_z, df_2d_gd, lrs_2d_marks[lr])

# NN

params_df = pd.read_parquet('dash_data/params_df.parquet')
params_df.sort_values(['step', 'strategy_name'], inplace=True)

graph_nn_all = dcc.Graph(
    figure=create_fig_nn_all(params_df, text='PCA for all strategies'),
    style={'width': '85vh', 'height': '80vh', 'display': 'inline-block'}
)

graph_nn_some = dcc.Graph(
    figure={},
    style={'width': '85vh', 'height': '80vh', 'display': 'inline-block'}
)

nn_strategy_dropdown = dcc.Dropdown(
    options=['A', 'B', 'C'],
    value='A', 
    clearable=False
)

@app.callback(
    Output(graph_nn_some, component_property='figure'),
    Input(nn_strategy_dropdown, component_property='value'),
)
def update_nn_strategy(strategy):
    # names = [f'strategy_n' for n in ['small', 'medium', 'large']]
    if strategy == 'A':
        names = ['A_small', 'A_medium', 'A_large',]
    elif strategy == 'B':
        names = ['B_small', 'B_medium', 'B_large',]
    elif strategy == 'C':
        names = ['C_small', 'C_medium', 'C_large',]

    return create_fig_nn_all(params_df[params_df['strategy_name'].isin(names)], text=f'PCA for strategy {strategy}')


losses_df = pd.read_parquet('dash_data/losses_df.parquet')
losses_df['train_loss'] /= 6

graph_nn_train_loss = dcc.Graph(
    figure=create_fig_nn_loss(losses_df, y='train_loss'),
    style={'width': '85vh', 'height': '80vh', 'display': 'inline-block'}
)

graph_nn_test_loss = dcc.Graph(
    figure=create_fig_nn_loss(losses_df, y='test_loss'),
    style={'width': '85vh', 'height': '80vh', 'display': 'inline-block'}
)

graph_nn_diff_loss = dcc.Graph(
    figure=create_fig_nn_diff_loss(losses_df),
)

# App

app.layout = dbc.Container([
    title, 
    html.Div(children=[
        graph_1d,
        graph_1d_ends
    ]),
    html.P("Learning rate"),
    lrs_1d_slider, 
    html.P("Start point"),
    start_points_1d_slider,
    html.Div(children=[
        graph_2d,
        graph_2d_ends
    ]),
    html.P("Learning rate"),
    lrs_2d_slider,
    html.Div(children=[
        graph_nn_all,
        graph_nn_some
    ]),
    nn_strategy_dropdown,
    html.Div(children=[
        graph_nn_train_loss,
        graph_nn_test_loss
    ]),
    graph_nn_diff_loss
])

if __name__=='__main__':
    app.run_server(port=8053)