import pandas as pd
import os
import pandas as pd

dataset_paths = ['datasets/ADA-USD.csv', 'datasets/AVAX-USD.csv', 'datasets/BCH-USD.csv', 'datasets/BNB-USD.csv', 'datasets/BTCB-USD.csv',
                 'datasets/BTC-USD.csv', 'datasets/CRO-USD.csv', 'datasets/DAI-USD.csv', 'datasets/DOGE-USD.csv', 'datasets/DOT-USD.csv',
                 'datasets/ETH-USD.csv', 'datasets/HBAR-USD.csv', 'datasets/ICP-USD.csv', 'datasets/LINK-USD.csv', 'datasets/LTC-USD.csv',
                 'datasets/MATIC-USD.csv', 'datasets/MXC-USD.csv', 'datasets/NEAR-USD.csv', 'datasets/SHIB-USD.csv', 'datasets/SOL-USD.csv',
                 'datasets/STX-USD.csv', 'datasets/TIA-USD.csv', 'datasets/TON-USD.csv', 'datasets/USDC-USD.csv', 'datasets/USDT-USD.csv',
                 'datasets/WETH-USD.csv', 'datasets/XEC-USD.csv', 'datasets/XLM-USD.csv', 'datasets/XMR-USD.csv', 'datasets/XRP-USD.csv', ]

crypto_symbols = ['ADA', 'AVAX', 'BCH', 'BNB', 'BTCB',
                  'BTC', 'CRO', 'DAI', 'DOGE', 'DOT',
                  'ETH', 'HBAR', 'ICP', 'LINK', 'LTC',
                  'MATIC', 'MXC', 'NEAR', 'SHIB', 'SOL',
                  'STX', 'TIA', 'TON', 'USDC', 'USDT',
                  'WETH', 'XEC', 'XLM', 'XMR', 'XRP', ]

files_and_symbols = [
    ("datasets/ADA-USD.csv", "ADA"),
    ("datasets/AVAX-USD.csv", "AVAX"),
    ("datasets/BCH-USD.csv", "BCH"),
    ("datasets/BNB-USD.csv", "BNB"),
    ("datasets/BTCB-USD.csv", "BTCB"),
    ("datasets/BTC-USD.csv", "BTC"),
    ("datasets/CRO-USD.csv", "CRO"),
    ("datasets/DAI-USD.csv", "DAI"),
    ("datasets/DOGE-USD.csv", "DOGE"),
    ("datasets/DOT-USD.csv", "DOT"),
    ("datasets/ETH-USD.csv", "ETH"),
    ("datasets/HBAR-USD.csv", "HBAR"),
    ("datasets/ICP-USD.csv", "ICP"),
    ("datasets/LINK-USD.csv", "LINK"),
    ("datasets/LTC-USD.csv", "LTC"),
    ("datasets/MATIC-USD.csv", "MATIC"),
    ("datasets/MXC-USD.csv", "MXC"),
    ("datasets/NEAR-USD.csv", "NEAR"),
    ("datasets/SHIB-USD.csv", "SHIB"),
    ("datasets/SOL-USD.csv", "SOL"),
    ("datasets/STX-USD.csv", "STX"),
    ("datasets/TIA-USD.csv", "TIA"),
    ("datasets/TON-USD.csv", "TON"),
    ("datasets/USDC-USD.csv", "USDC"),
    ("datasets/USDT-USD.csv", "USDT"),
    ("datasets/WETH-USD.csv", "WETH"),
    ("datasets/XEC-USD.csv", "XEC"),
    ("datasets/XLM-USD.csv", "XLM"),
    ("datasets/XMR-USD.csv", "XMR"),
    ("datasets/XRP-USD.csv", "XRP"),
]
all_dataframes = []

for file_path, symbol in files_and_symbols:
    df = pd.read_csv(file_path)
    df['Symbol'] = symbol
    all_dataframes.append(df)

merged_df = pd.concat(all_dataframes, ignore_index=True)
merged_df.sort_values(by='Date', inplace=True)
merged_df.to_csv("crypto.csv", index=False)
merged_df.head()