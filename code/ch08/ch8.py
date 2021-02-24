# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Financial Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

fn = '../../source/tr_eikon_eod_data.csv'

data = pd.read_csv(fn, index_col=0, parse_dates=True)

data.info()

data.head()

data.tail()

data.plot(figsize=(10, 12), subplots=True)

data.describe().round(2)

data.mean()

data.aggregate([min, np.mean, np.std, np.median, max])

# ## Changes Over Time

data.diff().head()

data.diff().mean()

# 相邻两行间的变化百分比：`pct_change()`

data.pct_change().round(3).head()

data.pct_change().mean().plot(kind='bar', figsize=(10,6))

rets = np.log(data / data.shift(1))

rets.head()

rets.cumsum().apply(np.exp).plot(figsize=(10, 6))

# ## Resampling

data.resample('1w', label='right').last().head()

data.resample('1m', label='right').last().head()

rets.cumsum().apply(np.exp).resample('1m', label='right').last().plot(figsize=(10,6))

# Compare above plot with the line with full data.

# ## Rolling Statistics

sym = 'AAPL.O'

rol_data = pd.DataFrame(data[sym]).dropna()

rol_data.tail()

window = 20

rol_data['min'] = rol_data[sym].rolling(window=window).min()
rol_data['mean'] = rol_data[sym].rolling(window=window).mean()
rol_data['std'] = rol_data[sym].rolling(window=window).std()
rol_data['median'] = rol_data[sym].rolling(window=window).median()
rol_data['min'] = rol_data[sym].rolling(window=window).min()
rol_data['max'] = rol_data[sym].rolling(window=window).max()
rol_data['ewma'] = rol_data[sym].ewm(halflife=0.5, min_periods=window).mean()

rol_data.dropna().head()

ax = rol_data[['min', 'mean', 'max']].iloc[-200:].plot(figsize=(10, 6), style=['g--', 'r--', 'm--'], lw=0.8);
rol_data[sym].iloc[-200:].plot(ax=ax, lw=2.0)

# 最后两句的语法比较奇怪。

# ### A Technical Analysis Example

rol_data['SMA1'] = rol_data[sym].rolling(window=42).mean()
rol_data['SMA2'] = rol_data[sym].rolling(window=252).mean()
rol_data[[sym, 'SMA1', 'SMA2']].tail()

rol_data[[sym, 'SMA1', 'SMA2']].plot(figsize=(10, 6))

tdata = rol_data.dropna()
tdata['positions'] = np.where(tdata['SMA1'] > tdata['SMA2'], 1, -1)

ax = tdata[[sym, 'SMA1', 'SMA2', 'positions']].plot(figsize=(10, 6), secondary_y='positions')
ax.get_legend().set_bbox_to_anchor((0.25, 0.85))

# 短期（42天）MA 高于长期 MA（252天）表明处于上升期，应买入（go long，看多？），反之处于下降期，应卖出（go short，看空？），这个规律很有意思。

# ## Correlation Analysis

# ## High-Frequency Data
