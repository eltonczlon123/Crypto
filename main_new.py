import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import timedelta
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import feedparser
from math import sqrt

data = pd.read_csv('crypto.csv')

data['Date'] = pd.to_datetime(data['Date'])
data_2021 = data[(data['Date'] >= '2017-11-11') & (data['Date'] <= '2024-04-07')]
pivot_data_2021 = data_2021.pivot_table(index='Symbol', columns='Date', values='Close', aggfunc='mean')
pivot_data_2021.shape, pivot_data_2021.head()
pivot_data_filled = pivot_data_2021.fillna(pivot_data_2021.mean(axis=0))
pca = PCA(n_components=10)
pca_transformed = pca.fit_transform(pivot_data_filled)


pca_df = pd.DataFrame(pca_transformed, index=pivot_data_filled.index, columns=[f"PC{i+1}" for i in range(10)])
print(pca_df.head())

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(pca_df)

pca_df['Cluster'] = clusters
print(pca_df.head(), pca_df['Cluster'].value_counts())

crypto_by_cluster = pca_df.reset_index().groupby('Cluster')['Symbol'].apply(list)

print(crypto_by_cluster)

selected_cryptos = ['BNB', 'BTC', 'ETH', 'ADA']
selected_data_2021 = pivot_data_filled.loc[selected_cryptos]

def correlation_analysis(selected_data):
    correlation_matrix = selected_data.transpose().corr()
    sorted_correlations = correlation_matrix.unstack().sort_values(kind="quicksort", ignore_index=False)
    high_correlations = sorted_correlations[sorted_correlations != 1].drop_duplicates()
    top_positive_correlations = high_correlations[-4:]
    top_negative_correlations = high_correlations[:4]
    top_positives = [(index[0] + '-' + index[1], val) for index, val in top_positive_correlations.items()]
    top_negatives = [(index[0] + '-' + index[1], val) for index, val in top_negative_correlations.items()]

    return top_positives, top_negatives


top_positives, top_negatives = correlation_analysis(selected_data_2021)
print("Top Positive Correlations:")
print(top_positives)
print("Top Negative Correlations:")
print(top_negatives)


bnb_data = selected_data_2021.loc['BNB']


def plot_daily_closing_prices_gui_bnb(data, crypto_name, master_frame):

    for widget in master_frame.winfo_children():
        widget.destroy()

    fig = Figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    ax.plot(data.index, data.values, label='Daily Close', color='blue')
    ax.set_title(f'Daily Closing Prices of {crypto_name} (2024)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True)
    mean_value = data.mean()
    median_value = data.median()
    std_dev = data.std()
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    stats_text = (f'Mean: {mean_value:.2f}\nMedian: {median_value:.2f}\n'
                  f'Std Dev: {std_dev:.2f}\nSkew: {skewness:.2f}\nKurtosis: {kurtosis:.2f}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    canvas = FigureCanvasTkAgg(fig, master=master_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def plot_histogram_bnb(data, crypto_name, master_frame):
    for widget in master_frame.winfo_children():
        widget.destroy()
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    sns.histplot(data, kde=True, color='blue', ax=ax)
    ax.set_title(f'Distribution of Daily Closing Prices of {crypto_name} (2024)')
    ax.set_xlabel('Price (USD)')
    ax.set_ylabel('Frequency')
    mean_value = data.mean()
    median_value = data.median()
    std_dev = data.std()
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    stats_text = (f'Mean: {mean_value:.2f}\nMedian: {median_value:.2f}\n'
                  f'Std Dev: {std_dev:.2f}\nSkew: {skewness:.2f}\nKurtosis: {kurtosis:.2f}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    canvas = FigureCanvasTkAgg(fig, master=master_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


bnb_returns = bnb_data.pct_change().dropna()


def plot_daily_returns_bnb(data, crypto_name, master_frame):
    daily_returns = data.pct_change().dropna()
    for widget in master_frame.winfo_children():
        widget.destroy()
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    sns.histplot(daily_returns, kde=True, color='blue', ax=ax)
    ax.set_title(f'Distribution of Daily Returns of {crypto_name} (2024)')
    ax.set_xlabel('Daily Returns')
    ax.set_ylabel('Frequency')
    mean_value = data.mean()
    median_value = data.median()
    std_dev = data.std()
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    stats_text = (f'Mean: {mean_value:.2f}\nMedian: {median_value:.2f}\n'
                  f'Std Dev: {std_dev:.2f}\nSkew: {skewness:.2f}\nKurtosis: {kurtosis:.2f}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    canvas = FigureCanvasTkAgg(fig, master=master_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

btc_data = selected_data_2021.loc['BTC']

def plot_daily_closing_prices_btc(data, crypto_name, master_frame):
    for widget in master_frame.winfo_children():
        widget.destroy()
    fig = Figure(figsize=(14, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(data.index, data.values, label='Daily Close', color='green')
    ax.set_title(f'Daily Closing Prices of {crypto_name} (2024)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True)
    mean_value = data.mean()
    median_value = data.median()
    std_dev = data.std()
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    stats_text = (f'Mean: {mean_value:.2f}\nMedian: {median_value:.2f}\n'
                  f'Std Dev: {std_dev:.2f}\nSkew: {skewness:.2f}\nKurtosis: {kurtosis:.2f}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    canvas = FigureCanvasTkAgg(fig, master=master_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def plot_histogram_btc(data, crypto_name, master_frame):
    for widget in master_frame.winfo_children():
        widget.destroy()
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    sns.histplot(data, kde=True, color='green', ax=ax)
    ax.set_title(f'Distribution of Daily Closing Prices of {crypto_name} (2024)')
    ax.set_xlabel('Price (USD)')
    ax.set_ylabel('Frequency')
    mean_value = data.mean()
    median_value = data.median()
    std_dev = data.std()
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    stats_text = (f'Mean: {mean_value:.2f}\nMedian: {median_value:.2f}\n'
                  f'Std Dev: {std_dev:.2f}\nSkew: {skewness:.2f}\nKurtosis: {kurtosis:.2f}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    canvas = FigureCanvasTkAgg(fig, master=master_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


btc_returns = btc_data.pct_change().dropna()


def plot_daily_returns_histogram(data, crypto_name, master_frame):
    daily_returns = data.pct_change().dropna()
    for widget in master_frame.winfo_children():
        widget.destroy()
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    sns.histplot(daily_returns, kde=True, color='green', ax=ax)
    ax.set_title(f'Distribution of Daily Returns of {crypto_name} (2024)')
    ax.set_xlabel('Daily Returns')
    ax.set_ylabel('Frequency')
    mean_value = data.mean()
    median_value = data.median()
    std_dev = data.std()
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    stats_text = (f'Mean: {mean_value:.2f}\nMedian: {median_value:.2f}\n'
                  f'Std Dev: {std_dev:.2f}\nSkew: {skewness:.2f}\nKurtosis: {kurtosis:.2f}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    canvas = FigureCanvasTkAgg(fig, master=master_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

eth_data = selected_data_2021.loc['ETH']


def plot_eth_daily_close(data, crypto_name, master_frame):
    for widget in master_frame.winfo_children():
        widget.destroy()
    fig = Figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    ax.plot(data.index, data.values, label='Daily Close', color='purple')
    ax.set_title(f'Daily Closing Prices of {crypto_name} (2024)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True)
    mean_value = data.mean()
    median_value = data.median()
    std_dev = data.std()
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    stats_text = (f'Mean: {mean_value:.2f}\nMedian: {median_value:.2f}\n'
                  f'Std Dev: {std_dev:.2f}\nSkew: {skewness:.2f}\nKurtosis: {kurtosis:.2f}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    canvas = FigureCanvasTkAgg(fig, master=master_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def plot_eth_price_distribution(data, crypto_name, master_frame):
    for widget in master_frame.winfo_children():
        widget.destroy()

    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    sns.histplot(data.values, kde=True, color='purple', ax=ax)
    ax.set_title(f'Distribution of Daily Closing Prices of {crypto_name} (2024)')
    ax.set_xlabel('Price (USD)')
    ax.set_ylabel('Frequency')

    mean_value = data.mean()
    median_value = data.median()
    std_dev = data.std()
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    stats_text = (f'Mean: {mean_value:.2f}\nMedian: {median_value:.2f}\n'
                  f'Std Dev: {std_dev:.2f}\nSkew: {skewness:.2f}\nKurtosis: {kurtosis:.2f}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    canvas = FigureCanvasTkAgg(fig, master=master_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


eth_returns = eth_data.pct_change().dropna()


def plot_eth_returns_distribution(data, crypto_name, master_frame):
    daily_returns = data.pct_change().dropna()
    for widget in master_frame.winfo_children():
        widget.destroy()
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    sns.histplot(daily_returns, kde=True, color='purple', ax=ax)
    ax.set_title(f'Distribution of Daily Returns of {crypto_name} (2024)')
    ax.set_xlabel('Daily Returns')
    ax.set_ylabel('Frequency')

    mean_value = data.mean()
    median_value = data.median()
    std_dev = data.std()
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    stats_text = (f'Mean: {mean_value:.2f}\nMedian: {median_value:.2f}\n'
                  f'Std Dev: {std_dev:.2f}\nSkew: {skewness:.2f}\nKurtosis: {kurtosis:.2f}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    canvas = FigureCanvasTkAgg(fig, master=master_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


ada_data = selected_data_2021.loc['ADA']

def plot_ada_daily_close(data, crypto_name, master_frame):
    for widget in master_frame.winfo_children():
        widget.destroy()

    fig = Figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    ax.plot(data.index, data.values, label='Daily Close', color='orange')
    ax.set_title(f'Daily Closing Prices of {crypto_name} (2024)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True)

    mean_value = data.mean()
    median_value = data.median()
    std_dev = data.std()
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    stats_text = (f'Mean: {mean_value:.2f}\nMedian: {median_value:.2f}\n'
                  f'Std Dev: {std_dev:.2f}\nSkew: {skewness:.2f}\nKurtosis: {kurtosis:.2f}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    canvas = FigureCanvasTkAgg(fig, master=master_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def plot_ada_price_distribution(data, crypto_name, master_frame):
    for widget in master_frame.winfo_children():
        widget.destroy()

    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    sns.histplot(data.values, kde=True, color='orange', ax=ax)
    ax.set_title(f'Distribution of Daily Closing Prices of {crypto_name} (2024)')
    ax.set_xlabel('Price (USD)')
    ax.set_ylabel('Frequency')

    mean_value = data.mean()
    median_value = data.median()
    std_dev = data.std()
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    stats_text = (f'Mean: {mean_value:.2f}\nMedian: {median_value:.2f}\n'
                  f'Std Dev: {std_dev:.2f}\nSkew: {skewness:.2f}\nKurtosis: {kurtosis:.2f}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    canvas = FigureCanvasTkAgg(fig, master=master_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


ada_returns = ada_data.pct_change().dropna()


def plot_ada_returns_distribution(data, crypto_name, master_frame):
    daily_returns = data.pct_change().dropna()
    for widget in master_frame.winfo_children():
        widget.destroy()
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    sns.histplot(daily_returns, kde=True, color='orange', ax=ax)
    ax.set_title(f'Distribution of Daily Returns of {crypto_name} (2024)')
    ax.set_xlabel('Daily Returns')
    ax.set_ylabel('Frequency')

    mean_value = data.mean()
    median_value = data.median()
    std_dev = data.std()
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    stats_text = (f'Mean: {mean_value:.2f}\nMedian: {median_value:.2f}\n'
                  f'Std Dev: {std_dev:.2f}\nSkew: {skewness:.2f}\nKurtosis: {kurtosis:.2f}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    canvas = FigureCanvasTkAgg(fig, master=master_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

cryptos = ['BTC', 'BNB', 'ADA', 'ETH']
scaled_data = {}
train_size = {}
X_train, X_test, y_train, y_test = {}, {}, {}, {}
svr_models = {}
svr_predictions = {}
trading_signals_svr = {}

for crypto in cryptos:
    data = selected_data_2021.loc[crypto].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data[crypto] = scaler.fit_transform(data)
    train_size[crypto] = int(len(scaled_data[crypto]) * 0.8)
    X_train[crypto] = scaled_data[crypto][:train_size[crypto], :]
    y_train[crypto] = scaled_data[crypto][:train_size[crypto], 0]
    X_test[crypto] = scaled_data[crypto][train_size[crypto]:, :]
    y_test[crypto] = scaled_data[crypto][train_size[crypto]:, 0]
    print(f"Shape of X_train[{crypto}]: {X_train[crypto].shape}")
    print(f"Shape of X_test[{crypto}]: {X_test[crypto].shape}")
    svr_model = SVR(kernel='rbf', C=1e3, gamma='scale', epsilon=0.1)
    svr_model.fit(X_train[crypto], y_train[crypto])
    svr_models[crypto] = svr_model
    svr_predictions[crypto] = svr_model.predict(X_test[crypto])
    svr_predictions[crypto] = scaler.inverse_transform(svr_predictions[crypto].reshape(-1, 1))

    signals = []
    num_days = 31
    if len(svr_predictions[crypto]) > 1:
        for i in range(1, num_days):
            signal = 'Buy' if svr_predictions[crypto][i] > svr_predictions[crypto][i - 1] else 'Sell'
            signals.append(signal)
        trading_signals_svr[crypto] = {
            'Dates': pd.date_range(start=pd.to_datetime('2024-04-08'), periods=num_days - 1, freq='D'),
            'Signals': signals
        }

for crypto, forecast in svr_predictions.items():
    print(f"SVR Predictions for {crypto}:")
    start_date = pd.to_datetime('2024-04-08')
    end_date = pd.to_datetime('2024-05-08')
    num_days = (end_date - start_date).days + 1
    for date, price in zip(pd.date_range(start=start_date, periods=num_days, freq='D'), forecast[:num_days]):
        print(f"{date.date()} - {price[0]:.6f}")

cryptos = ['BTC', 'BNB', 'ADA', 'ETH']
prophet_results = {}
prophet_models = {}
prophet_signals = {}

start_date = pd.to_datetime('2024-04-07')
end_date = pd.to_datetime('2024-05-07')

for crypto in cryptos:
    data_prophet = pd.DataFrame({
        'ds': pd.to_datetime(selected_data_2021.columns),
        'y': selected_data_2021.loc[crypto].values
    })
    train_data = data_prophet[data_prophet['ds'] < start_date]

    model_prophet = Prophet(daily_seasonality=True)
    model_prophet.fit(train_data)

    future_prophet = model_prophet.make_future_dataframe(periods=30, freq='D', include_history=False)
    future_prophet = future_prophet[future_prophet['ds'] >= start_date]

    forecast_prophet = model_prophet.predict(future_prophet)

    prophet_results[crypto] = forecast_prophet[['ds', 'yhat']]
    actual_prices = data_prophet[data_prophet['ds'].isin(forecast_prophet['ds'])]['y'].values
    forecasted_prices = forecast_prophet['yhat'].values

    signals = []
    for i in range(1, len(forecasted_prices)):
        signals.append('Buy' if forecasted_prices[i] > forecasted_prices[i - 1] else 'Sell')
    prophet_signals[crypto] = {
        'Dates': forecast_prophet['ds'][1:],
        'Signals': signals
    }

print("Prophet Predictions for BTC:")
print(prophet_results['BTC'])
print("\n")

print("Prophet Predictions for ETH:")
print(prophet_results['ETH'])
print("\n")

print("Prophet Predictions for ADA:")
print(prophet_results['ADA'])
print("\n")

print("Prophet Predictions for BNB:")
print(prophet_results['BNB'])
print("\n")

btc_data_arima = selected_data_2021.loc['BTC'].dropna()
result_adf = adfuller(btc_data_arima)
adf_statistic = result_adf[0]
p_value = result_adf[1]
critical_values = result_adf[4]

print(adf_statistic)
print(p_value)
print(critical_values)

btc_data_diff = btc_data_arima.diff().dropna()

result_adf_diff = adfuller(btc_data_diff)
adf_statistic_diff = result_adf_diff[0]
p_value_diff = result_adf_diff[1]

print(adf_statistic_diff)
print(p_value_diff)

arima_metrics = {}
arima_results = {}
cryptos_to_forecast = ['BNB', 'ADA', 'ETH', 'BTC']
arima_signals = {}

actual_future_data = {crypto: selected_data_2021.loc[crypto].dropna().tail(30) for crypto in cryptos_to_forecast}

for crypto in cryptos_to_forecast:
    crypto_data = selected_data_2021.loc[crypto].dropna()[:-30]
    model_arima = ARIMA(crypto_data, order=(0, 2, 2))
    fitted_model = model_arima.fit()
    forecast_result = fitted_model.get_forecast(steps=30)
    forecast_mean = forecast_result.predicted_mean

    arima_results[crypto] = {
        '(ARIMA)Forecast Mean': forecast_mean,
        'Confidence Interval': forecast_result.conf_int()
    }
    actual_prices = actual_future_data[crypto]
    mae = mean_absolute_error(actual_prices, forecast_mean)
    mse = mean_squared_error(actual_prices, forecast_mean)
    rmse = np.sqrt(mse)
    print(f"(ARIMA) Metrics for {crypto}: MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}\n")
    signals = []
    forecasted_prices = forecast_mean.values

    for i in range(1, len(forecasted_prices)):
        signals.append('Buy' if forecasted_prices[i] > forecasted_prices[i - 1] else 'Sell')
    arima_signals[crypto] = {
        'Dates': pd.date_range(start=start_date, periods=len(forecasted_prices), freq='D')[1:],
        'Signals': signals
    }

print(arima_results)


def create_lagged_features(data, n_lags=30):
    X, y = [], []
    if len(data) >= n_lags:
        for i in range(n_lags, len(data)):
            X.append(data[i - n_lags:i])
            y.append(data[i])
    return np.array(X), np.array(y)


def perform_rolling_forecast(model, initial_data, days_forecast=30):
    forecasted_prices = []
    current_features = initial_data.copy()

    for _ in range(days_forecast):
        next_day_prediction = model.predict(current_features.reshape(1, -1))
        forecasted_prices.append(next_day_prediction[0])

        current_features = np.roll(current_features, -1)
        current_features[-1] = next_day_prediction[0]

    return forecasted_prices


cryptos = ['BTC', 'BNB', 'ADA', 'ETH']
models = {}
results = {}
rolling_forecast_results = {}

for crypto in cryptos:
    data = selected_data_2021.loc[crypto].dropna().values.flatten()
    transformed_data = np.log(data + 0.0001)  # Smoothing data to stabilize variance

    X, y = create_lagged_features(transformed_data)
    if len(X) > 0:
        # Splitting the data into training and testing segments
        X_train, X_test = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]
        y_train, y_test = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]

        # Random Forest model training
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Predicting on test data
        y_pred = rf_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        models[crypto] = rf_model
        results[crypto] = {'RMSE': rmse}

        # Performing rolling forecast using the last known data
        last_known_features = X_test[-1]
        forecasts = perform_rolling_forecast(rf_model, last_known_features, 30)
        last_known_date = pd.to_datetime(selected_data_2021.columns[-1])
        forecast_dates = pd.date_range(start=last_known_date + timedelta(days=1), periods=30, freq='D')

        forecast_prices = np.exp(forecasts) - 0.0001  # Inverting log transformation

        rolling_forecast_results[crypto] = {
            'Dates': forecast_dates,
            'Forecasted Prices': forecast_prices
        }

# Output the forecasting results and RMSE metrics
for crypto, info in rolling_forecast_results.items():
    print(f"Random Forest Forecast for {crypto}:")
    for date, price in zip(info['Dates'], info['Forecasted Prices']):
        print(f"{date.date()} - {price:.6f}")
    print("\n")
    print(f"RMSE for {crypto}: {results[crypto]['RMSE']:.4f}\n")

def generate_trading_signals(forecasted_prices):
    signals = []
    for i in range(1, len(forecasted_prices)):
        if forecasted_prices[i] > forecasted_prices[i - 1]:
            signals.append('Buy')
        else:
            signals.append('Sell')
    return signals


trading_signals_rf = {}
for crypto, info in rolling_forecast_results.items():
    signals = generate_trading_signals(info['Forecasted Prices'])
    trading_signals_rf[crypto] = {
        'Dates': info['Dates'][1:],
        'Signals': signals
    }

for crypto, info in trading_signals_rf.items():
    print(f"(Random Forest) Trading Signals for {crypto}:")
    for date, signal in zip(info['Dates'], info['Signals']):
        print(f"{date.date()} - {signal}")
    print("\n")

for crypto, info in trading_signals_svr.items():
    print(f"(SVR) Trading Signals for {crypto}:")
    for date, signal in zip(info['Dates'], info['Signals']):
        print(f"{date.date()} - {signal}")
    print("\n")

for crypto, info in prophet_signals.items():
    print(f"(Prophet) Trading Signals for {crypto}:")
    for date, signal in zip(info['Dates'], info['Signals']):
        if date <= end_date:
            print(f"{date.date()} - {signal}")
    print("\n")

for crypto, info in arima_signals.items():
    print(f"(ARIMA) Trading Signals for {crypto}:")
    for date, signal in zip(info['Dates'], info['Signals']):
        if date <= end_date:
            print(f"{date.date()} - {signal}")
    print("\n")


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_scaled_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_pred - y_true)) / np.mean(np.abs(y_true[1:] - y_true[:-1]))


for crypto in cryptos:

    y_true = selected_data_2021.loc[crypto].dropna().values.flatten()[train_size[crypto]:]
    y_pred = svr_predictions[crypto].flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)

    print(f"SVR metrics for {crypto}:")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}")
    print(f"MASE: {mase}")
    print("\n")

root = tk.Tk()
root.title('Cryptocurrency Analysis Toolkit')


tabControl = ttk.Notebook(root)

def display_bnb_closing_prices():
    bnb_data = selected_data_2021.loc['BNB']
    plot_daily_closing_prices_gui_bnb(bnb_data, 'BNB', eda_plot_frame)

def display_bnb_histogram():
    bnb_data = selected_data_2021.loc['BNB']
    plot_histogram_bnb(bnb_data, 'BNB', eda_plot_frame)

def display_bnb_returns():
    bnb_data = selected_data_2021.loc['BNB']
    plot_daily_returns_bnb(bnb_data, 'BNB', eda_plot_frame)

def display_btc_closing_prices():
    btc_data = selected_data_2021.loc['BTC']
    plot_daily_closing_prices_btc(btc_data, 'BTC', eda_plot_frame)

def display_btc_histogram():
    btc_data = selected_data_2021.loc['BTC']
    plot_histogram_btc(btc_data, 'BTC', eda_plot_frame)

def display_btc_returns():
    btc_data = selected_data_2021.loc['BTC']
    plot_daily_returns_histogram(btc_data, 'BTC', eda_plot_frame)

def display_eth_daily():
    eth_data = selected_data_2021.loc['ETH']
    plot_eth_daily_close(eth_data, 'ETH', eda_plot_frame)

def display_eth_distrubtion():
    eth_data = selected_data_2021.loc['ETH']
    plot_eth_price_distribution(eth_data, 'ETH', eda_plot_frame)

def display_eth_returns_distrubtion():
    eth_data = selected_data_2021.loc['ETH']
    plot_eth_returns_distribution(eth_data, 'ETH', eda_plot_frame)

def display_ada_daily_close():
    ada_data = selected_data_2021.loc['ADA']
    plot_ada_daily_close(ada_data, 'ADA', eda_plot_frame)

def display_ada_price_distribution():
    ada_data = selected_data_2021.loc['ADA']
    plot_ada_price_distribution(ada_data, 'ADA', eda_plot_frame)

def display_ada_returns_distrubtion():
    ada_data = selected_data_2021.loc['ADA']
    plot_ada_returns_distribution(ada_data, 'ADA', eda_plot_frame)

def create_correlation_treeview(parent):
    container = ttk.Frame(parent)
    canvas = tk.Canvas(container)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.grid(row=0, column=0, sticky='nsew')
    scrollbar.grid(row=0, column=1, sticky='ns')
    container.grid(sticky='nsew')

    tree = ttk.Treeview(scrollable_frame, columns=('Crypto Pair', 'Correlation Value'), show='headings')
    tree.heading('Crypto Pair', text='Crypto Pair')
    tree.heading('Correlation Value', text='Correlation Value')
    tree.pack(expand=True, fill='both')

    return tree

def update_correlation_display():
    top_positives, top_negatives = correlation_analysis(selected_data_2021)
    for i in correlation_tree.get_children():
        correlation_tree.delete(i)
    correlation_tree.insert('', 'end', values=("Top Positive Correlations", ""))
    for pair, value in top_positives:
        correlation_tree.insert('', 'end', values=(pair, f"{value:.2f}"))
    correlation_tree.insert('', 'end', values=("Top Negative Correlations", ""))
    for pair, value in top_negatives:
        correlation_tree.insert('', 'end', values=(pair, f"{value:.2f}"))


def display_crypto_stats():
    crypto = selected_crypto_corr.get()
    if crypto in selected_data_2021.index:
        crypto_data = selected_data_2021.loc[crypto]
        mean_price = crypto_data.mean()
        median_price = crypto_data.median()
        mode_price = stats.mode(crypto_data)[0][0]
        std_dev = crypto_data.std()
        stats_text = f"Statistics for {crypto}:\n\n" \
                     f"Mean Price: {mean_price}\n" \
                     f"Median Price: {median_price}\n" \
                     f"Mode Price: {mode_price}\n" \
                     f"Standard Deviation: {std_dev}"
        for widget in corr_frame.winfo_children():
            if isinstance(widget, ttk.Label):
                widget.destroy()

        stats_label = ttk.Label(corr_frame, text=stats_text)
        stats_label.grid(column=2, row=0, padx=10, pady=10)


def display_svr_forecast():
    crypto = selected_crypto_forecast.get()
    forecast = svr_predictions[crypto]
    forecast_window = tk.Toplevel(root)
    forecast_window.title(f"SVR Forecast for {crypto}")


    text_widget = tk.Text(forecast_window, wrap='word')
    text_widget.pack(padx=10, pady=10)


    for date, price in zip(pd.date_range(start=pd.to_datetime('2024-04-08'), periods=len(forecast), freq='D'),
                           forecast):
        text_widget.insert('end', f"{date.date()} - {price[0]:.6f}\n")


def display_prophet_forecast():
    crypto = selected_crypto_forecast.get()
    forecast = prophet_results[crypto]
    forecast_window = tk.Toplevel(root)
    forecast_window.title(f"Prophet Forecast for {crypto}")
    text_widget = tk.Text(forecast_window, wrap='word')
    text_widget.pack(padx=10, pady=10)
    for index, row in forecast.iterrows():
        text_widget.insert('end', f"{row['ds'].date()} - {row['yhat']:.6f}\n")


def display_arima_forecast():
    crypto = selected_crypto_forecast.get()
    forecast = arima_results[crypto]['(ARIMA)Forecast Mean']
    forecast_window = tk.Toplevel(root)
    forecast_window.title(f"ARIMA Forecast for {crypto}")

    text_widget = tk.Text(forecast_window, wrap='word')
    text_widget.pack(padx=10, pady=10)


    for date, price in zip(forecast.index, forecast.values):
        text_widget.insert('end', f"{date.date()} - {price:.6f}\n")


def display_rf_forecast():
    crypto = selected_crypto_forecast.get()
    forecast = rolling_forecast_results[crypto]['Forecasted Prices']
    forecast_dates = rolling_forecast_results[crypto]['Dates']
    forecast_window = tk.Toplevel(root)
    forecast_window.title(f"Random Forest Forecast for {crypto}")
    text_widget = tk.Text(forecast_window, wrap='word')
    text_widget.pack(padx=10, pady=10)
    for date, price in zip(forecast_dates, forecast):
        text_widget.insert('end', f"{date.date()} - {price:.6f}\n")


def display_svr_signals():
    crypto = selected_crypto_signals.get()
    signals_data = trading_signals_svr[crypto]
    signals_window = tk.Toplevel(root)
    signals_window.title(f"SVR Trading Signals for {crypto}")
    text_widget = tk.Text(signals_window, wrap='word')
    text_widget.pack(padx=10, pady=10)
    for date, signal in zip(signals_data['Dates'], signals_data['Signals']):
        text_widget.insert('end', f"{date.date()} - {signal}\n")


def display_prophet_signals():
    crypto = selected_crypto_signals.get()
    signals_data = prophet_signals[crypto]
    signals_window = tk.Toplevel(root)
    signals_window.title(f"Prophet Trading Signals for {crypto}")
    text_widget = tk.Text(signals_window, wrap='word')
    text_widget.pack(padx=10, pady=10)

    for date, signal in zip(signals_data['Dates'], signals_data['Signals']):
        text_widget.insert('end', f"{date.date()} - {signal}\n")


def display_arima_signals():
    crypto = selected_crypto_signals.get()
    signals_data = arima_signals[crypto]
    signals_window = tk.Toplevel(root)
    signals_window.title(f"ARIMA Trading Signals for {crypto}")
    text_widget = tk.Text(signals_window, wrap='word')
    text_widget.pack(padx=10, pady=10)
    for date, signal in zip(signals_data['Dates'], signals_data['Signals']):
        text_widget.insert('end', f"{date.date()} - {signal}\n")


def display_rf_signals():
    crypto = selected_crypto_signals.get()
    signals_data = trading_signals_rf[crypto]
    signals_window = tk.Toplevel(root)
    signals_window.title(f"Random Forest Trading Signals for {crypto}")
    text_widget = tk.Text(signals_window, wrap='word')
    text_widget.pack(padx=10, pady=10)
    for date, signal in zip(signals_data['Dates'], signals_data['Signals']):
        text_widget.insert('end', f"{date.date()} - {signal}\n")


def display_news():
    news_feed = feedparser.parse('https://www.coindesk.com/arc/outboundfeeds/rss/')
    news_window = tk.Toplevel(root)
    news_window.title("Crypto News")
    text_widget = tk.Text(news_window, wrap='word')
    text_widget.pack(padx=10, pady=10)
    for entry in news_feed.entries:
        text_widget.insert('end', f"{entry.title}\n{entry.link}\n\n")

tabs = ['EDA', 'Correlation', 'Forecasting', 'Signals', 'News']
crypto_names = ['BTC', 'ETH', 'BNB', 'ADA']

tab_frames = {tab: ttk.Frame(tabControl) for tab in tabs}
for tab, frame in tab_frames.items():
    tabControl.add(frame, text=tab)

tabControl.pack(expand=1, fill="both")

eda_frame = tab_frames['EDA']
ttk.Label(eda_frame, text="Select Data to Display:").grid(column=0, row=0)
ttk.Button(eda_frame, text="Closing Prices BNB Data", command=display_bnb_closing_prices).grid(column=1, row=0, padx=10, pady=10)
ttk.Button(eda_frame, text="Histogram BNB Data", command=display_bnb_histogram).grid(column=2, row=0, padx=10, pady=10)
ttk.Button(eda_frame, text="Daily Returns BNB Data", command=display_bnb_returns).grid(column=3, row=0, padx=10, pady=10)

ttk.Button(eda_frame, text="Closing Prices BTC Data", command=display_btc_closing_prices).grid(column=1, row=10, padx=10, pady=10)
ttk.Button(eda_frame, text="Histogram BTC Data", command=display_btc_histogram).grid(column=2, row=10, padx=10, pady=10)
ttk.Button(eda_frame, text="Daily Returns BTC Data", command=display_btc_returns).grid(column=3, row=10, padx=10, pady=10)

ttk.Button(eda_frame, text="Closing Prices ETH Data", command=display_eth_daily).grid(column=1, row=20, padx=10, pady=10)
ttk.Button(eda_frame, text="Histogram ETH Data", command=display_eth_distrubtion).grid(column=2, row=20, padx=10, pady=10)
ttk.Button(eda_frame, text="Daily Returns ETH Data", command=display_eth_returns_distrubtion).grid(column=3, row=20, padx=10, pady=10)

ttk.Button(eda_frame, text="Closing Prices ADA Data", command=display_ada_daily_close).grid(column=1, row=30, padx=10, pady=10)
ttk.Button(eda_frame, text="Histogram ADA Data", command=display_ada_price_distribution).grid(column=2, row=30, padx=10, pady=10)
ttk.Button(eda_frame, text="Daily Returns ADA Data", command=display_ada_returns_distrubtion).grid(column=3, row=30, padx=10, pady=10)

corr_frame = tab_frames['Correlation']
correlation_tree = create_correlation_treeview(corr_frame)
ttk.Button(corr_frame, text="Analyze Correlations", command=update_correlation_display).grid(column=0, row=1, padx=10, pady=10)
ttk.Button(corr_frame, text="Display Statistics", command=display_crypto_stats).grid(column=1, row=1, padx=10, pady=10)

selected_crypto_corr = tk.StringVar()
dropdown_corr = ttk.Combobox(corr_frame, textvariable=selected_crypto_corr)
dropdown_corr['values'] = crypto_names
dropdown_corr.grid(column=4, row=4, padx=10, pady=10)

corr_frame.grid_rowconfigure(2, weight=1)

stats_frame = tk.Frame(corr_frame)
stats_frame.grid(row=2, column=0, columnspan=3, sticky='nsew')

eda_plot_frame = tk.Frame(eda_frame)
eda_plot_frame.grid(column=0, row=1, columnspan=2, sticky='nsew')
eda_frame.columnconfigure(0, weight=1)
eda_frame.rowconfigure(1, weight=1)

forecasting_frame = tab_frames['Forecasting']
selected_crypto_forecast = tk.StringVar()
dropdown_forecast = ttk.Combobox(forecasting_frame, textvariable=selected_crypto_forecast)
dropdown_forecast['values'] = crypto_names
dropdown_forecast.grid(column=0, row=0, padx=10, pady=10)

ttk.Button(forecasting_frame, text="Display SVR Forecast", command=display_svr_forecast).grid(column=1, row=0, padx=10, pady=10)
ttk.Button(forecasting_frame, text="Display Prophet Forecast", command=display_prophet_forecast).grid(column=2, row=0, padx=10, pady=10)
ttk.Button(forecasting_frame, text="Display ARIMA Forecast", command=display_arima_forecast).grid(column=5, row=0, padx=10, pady=10)
ttk.Button(forecasting_frame, text="Display Random Forest Forecast", command=display_rf_forecast).grid(column=7, row=0, padx=10, pady=10)

signals_frame = tab_frames['Signals']
selected_crypto_signals = tk.StringVar()
dropdown_signals = ttk.Combobox(signals_frame, textvariable=selected_crypto_signals)
dropdown_signals['values'] = crypto_names
dropdown_signals.grid(column=0, row=0, padx=10, pady=10)
ttk.Button(signals_frame, text="Display SVR Signals", command=display_svr_signals).grid(column=1, row=0, padx=10, pady=10)
ttk.Button(signals_frame, text="Display Prophet Signals", command=display_prophet_signals).grid(column=2, row=0, padx=10, pady=10)
ttk.Button(signals_frame, text="Display ARIMA Signals", command=display_arima_signals).grid(column=3, row=0, padx=10, pady=10)
ttk.Button(signals_frame, text="Display Random Forest Signals", command=display_rf_signals).grid(column=4, row=0, padx=10, pady=10)

news_frame = tab_frames['News']
ttk.Button(news_frame, text="Display News", command=display_news).pack()


root.mainloop()
