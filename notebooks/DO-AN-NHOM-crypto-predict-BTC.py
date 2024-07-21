# pip install dash pandas plotly scikit-learn xgboost tensorflow
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import pandas as pd
import numpy as np
import math
import datetime as dt

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from itertools import cycle
import plotly.graph_objects as go # add details
import plotly.express as px
from plotly.subplots import make_subplots

import seaborn as sns 
import matplotlib.pyplot as plt 
from colorama import Fore #for adding colours

# Load the data
df = pd.read_csv('../data/BTC-USD.csv')
df['Date'] = pd.to_datetime(df['Date'])
df=df.dropna()
df.set_index('Date', inplace=True)

# Prepare the data
scaler = MinMaxScaler()
df['Scaled_Close'] = scaler.fit_transform(df[['Close']])

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

sequence_length = 10
X, y = create_sequences(df['Scaled_Close'].values, sequence_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# LSTM model
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Create the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Bitcoin Price Prediction Dashboard'),
    
    dcc.Dropdown(
        id='model-selector',
        options=[
            {'label': 'XGBoost', 'value': 'xgboost'},
            {'label': 'LSTM', 'value': 'lstm'}
        ],
        value='xgboost'
    ),
    
    dcc.Graph(id='price-chart'),
    
    html.Div(id='performance-metrics')
])

@app.callback(
    [Output('price-chart', 'figure'),
     Output('performance-metrics', 'children')],
    [Input('model-selector', 'value')]
)
def update_chart(selected_model):
    if selected_model == 'xgboost':
        predictions = xgb_model.predict(X_test.reshape(X_test.shape[0], -1))
    else:
        predictions = lstm_model.predict(X_test)
    
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-len(actual):], y=actual, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=df.index[-len(predictions):], y=predictions, mode='lines', name='Predicted'))
    fig.update_layout(title='Bitcoin Price - Actual vs Predicted', xaxis_title='Date', yaxis_title='Price')
    
    mse = np.mean((predictions - actual)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actual))
    
    metrics = [
        html.P(f"Mean Squared Error: {mse:.2f}"),
        html.P(f"Root Mean Squared Error: {rmse:.2f}"),
        html.P(f"Mean Absolute Error: {mae:.2f}")
    ]
    
    return fig, metrics

if __name__ == '__main__':
    app.run_server(debug=True)