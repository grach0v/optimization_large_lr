from dash import Dash, dcc, Output, Input, html 
import dash_bootstrap_components as dbc   
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

def create_fig_1d(lr, start_point, df_1d_toy):

    fig_line = px.line(
        data_frame=df_1d_toy.query('type == "func"'), 
        x='x', 
        y='y'
    )

    fig_scatter = px.scatter(
        data_frame=df_1d_toy\
            .query('type == "func"')\
            .sort_values('y')\
            .groupby('minima_type', as_index=False)\
            .first()\
            .sort_values('minima_type'),
        x='x',
        y='y',
        color='minima_type', 
        symbol='minima_type',
        symbol_sequence=['square', 'diamond', 'x']
    )

    fig_scatter.update_traces(marker_size=12)

    x_min_func = df_1d_toy.query('type == "func"')['x'].min()
    x_max_func = df_1d_toy.query('type == "func"')['x'].max()

    fig_scatter_gd = px.scatter(
        data_frame=df_1d_toy.query(
            '''
            (type == "gd") & (x >= @x_min_func) & (x <= @x_max_func) & (lr == @lr) & (start_point == @start_point)
            '''
        ),
        x='x',
        y='y',
        color='step'
    )#.update_traces(mode="lines+markers")

    fig_scatter_gd.update_traces(marker_size=8)


    fig_1d = go.Figure(data=fig_line.data + fig_scatter.data + fig_scatter_gd.data)

    fig_1d.update_layout(
        title={
            'text': f'1D Toy example with lr={lr:.4f} x_start={start_point:.2f}', 
            'x':0.5,
            'xanchor': 'center',
        }
    )

    fig_1d.update(layout_coloraxis_showscale=False)

    return fig_1d

def create_fig_1d_ends(df_1d_toy):

    fig_ends = px.scatter(
        df_1d_toy\
            .query('type == "gd"')\
            .sort_values('step')\
            .groupby(['lr', 'start_point'], as_index=False)\
            .last()\
            .sort_values('minima_type'),
        x='start_point',
        y='lr',
        color='minima_type',
        symbol='minima_type',
        log_y=True,
        symbol_sequence=['square', 'diamond', 'x']

    ).update_layout(
        title={
            'text': 'Type of obtained minima', 
            'x':0.5,
            'xanchor': 'center',
        },        
    ).update_traces(marker_size=8)

    return fig_ends

# 2D

def create_fig_2d(grid_x, grid_y, grid_z, df_2d_toy, lr):
    df = df_2d_toy.query('lr == @lr')

    fig = go.Figure(
        data=[
            go.Surface(
                x=grid_x, 
                y=grid_y, 
                z=grid_z, 
                colorscale='Viridis', 
                showscale=False
            ),
            go.Scatter3d(
                x=df['x'], y=df['y'], z=df['z'], marker={'size': 3}
            )
        ],
    ).update_layout(
        title={
            'text': f'GD converged to flat minima lr={lr:.2E}', 
            'x':0.5,
            'xanchor': 'center',
        }
    )

    return fig

def create_fig_2d_ends(ends_counts):
    fig = px.line(
        data_frame=ends_counts.sort_values('lr'), 
        x='lr', 
        y='fraction', 
        color='type_minima', 
        line_dash='type_minima'
    ).update_layout(
        title={
            'text': 'fraction of types of minima obtained', 
            'x':0.5,
            'xanchor': 'center',
        }
    )

    return fig

def create_fig_nn_all(params_df, text):
    pca = PCA(n_components=3)
    X = pca.fit_transform(params_df.drop(['lr', 'step', 'strategy_name'], axis=1))
    start_id_ind = params_df['step'] == 1

    fig_line = px.line(x=X[:, 0], y=X[:, 1], color=params_df['strategy_name'])
    fig_line.update_traces(textposition="bottom right")

    fig_start = px.scatter(x=X[start_id_ind, 0], y=X[start_id_ind, 1]).update_traces(marker_size=12)

    fig_params = go.Figure(data=fig_line.data + fig_start.data)

    fig_params.update_layout(
        title={
            'text': text, 
            'x':0.5,
            'xanchor': 'center',
        }
    )    

    return fig_params

def create_fig_nn_loss(losses_df, y):
    fig = px.line(losses_df, x='step_normed', y=y, color='strategy_name')
    return fig

def create_fig_nn_diff_loss(losses_df):
    losses_df['diff_loss'] = losses_df['train_loss'] - losses_df['test_loss']
    fig = px.line(losses_df, x='step_normed', y='diff_loss', color='strategy_name')
    return fig

