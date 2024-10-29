# %% [markdown]
# Analysis of CO2 Emissions from Fuel Sources

# %% [markdown]
# This notebook explores CO2 emissions trends by fuel source, focusing on natural gas emissions and their environmental impact. The analysis includes data preprocessing, visualization, and forecasting using time series models.

# %% [markdown]
# Import Required Libraries

# %% [markdown]
# We will import the necessary libraries for data manipulation, visualization, and time series analysis.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
from pmdarima import auto_arima
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error

# %% [markdown]
# Initial Exploration of Data

# %% [markdown]
# We will read the dataset and perform an initial exploration to understand its structure and content.

# %%
data = pd.read_csv('CO2_T12_06.csv')
data.head()
data.tail()
data.info()

# %% [markdown]
# Data Overview:<br>
# - No missing values are present based on initial exploration.<br>
# - The `Value` column is in string format and needs to be converted to numeric for analysis.

# %%
data.describe()
data.describe(include='O')

# %% [markdown]
# Preprocessing Steps:<br>
# Let's ensure all data is numeric and time series formatted, then remove any rows with conversion issues.<br>
# Converting 'Value' to a numeric type, and handling dates

# %%
data['Value'] = pd.to_numeric(data['Value'], errors='coerce')
data['YYYYMM'] = pd.to_datetime(data['YYYYMM'], format='%Y%m', errors='coerce')
data = data.dropna()

# %% [markdown]
# Key Question 1:<br>
# What fuel sources contribute the most to CO2 emissions, and how do emissions trends change over time?

# %%
def plot_emissions_by_fuel(data):
    """
    Plots CO2 emissions trends by fuel type over time.
    Parameters:
    - data (pd.DataFrame): Dataset to plot.
    Returns:
    - None, displays a plot.
    """
    fuels = data.groupby('Description')
    fig, ax = plt.subplots(figsize=(14,8))
    for desc, group in fuels:
        ax.plot(group['YYYYMM'], group['Value'], label=desc)
    ax.set_title('CO2 Emissions by Fuel Type')
    ax.set_xlabel('Date')
    ax.set_ylabel('Emissions (Value)')
    plt.legend()
    plt.show()

# %%
plot_emissions_by_fuel(data)

# %% [markdown]
# Summing emissions by fuel type

# %%
total_emissions_by_fuel = data.groupby('Description')['Value'].sum()
total_emissions_by_fuel

# %%
def plot_total_emissions(emissions_summary):
    """
    Plots total CO2 emissions by fuel type as a bar chart.
    Parameters:
    - emissions_summary (pd.Series): Total emissions for each fuel type.
    Returns:
    - None, displays a bar plot.
    """
    plt.figure(figsize=(10,6))
    emissions_summary.plot(kind='bar', color='teal', title='Total CO2 Emissions by Fuel')
    plt.ylabel("Total Emissions")
    plt.xticks(rotation="vertical")
    plt.show()

# %%
plot_total_emissions(total_emissions_by_fuel)

# %% [markdown]
# Answer to Question 1:<br>
# Based on the charts, `coal` and `natural gas` are the most significant contributors to CO2 emissions. These trends likely result from their widespread use in power generation and industry.

# %% [markdown]
# Converting Data to Time Series Format

# %%
ts_data = data.set_index('YYYYMM')
ts_data.head()

# %% [markdown]
# Seasonal Decomposition of Emissions<br>
# We will investigate the seasonal patterns in CO2 emissions over time.<br>
# Grouping emissions data by month and fuel type

# %%
monthly_emissions = ts_data.groupby([pd.Grouper(freq='M'), 'Description'])['Value'].sum().unstack(level=1)
monthly_emissions.head()

# %% [markdown]
# Question 2:<br>
# Investigate the role of natural gas in CO2 emissions and its environmental impact.<br>
# Extracting natural gas emissions for analysis

# %%
natgas_emissions = monthly_emissions['Natural Gas Electric Power Sector CO2 Emissions']

# %% [markdown]
# Seasonal decomposition

# %%
decomposed = seasonal_decompose(natgas_emissions, model='multiplicative', period=12)
decomposed.plot()
plt.show()

# %% [markdown]
# Analysis Summary:<br>
# - The decomposition reveals a clear trend, seasonal patterns, and residual components for natural gas emissions, suggesting periodic fluctuations with an overall increase in emissions.

# %% [markdown]
# Model Building and Forecasting<br>
# We will build a forecasting model to predict future natural gas emissions.

# %% [markdown]
# Splitting data into train and test sets for model training

# %%
train_data = natgas_emissions['1973':'2014']
test_data = natgas_emissions['2015':]

# %%
# Extracting natural gas emissions for analysis as a DataFrame
natgas_emissions = monthly_emissions[['Natural Gas Electric Power Sector CO2 Emissions']]
natgas_emissions.columns = ['Value']  # Renaming for consistency

# Splitting data into train and test sets for model training
train_data = natgas_emissions['1973':'2014']
test_data = natgas_emissions['2015':]
test_data = test_data.reset_index()  # Reset index to get YYYYMM as a column

# %% [markdown]
# Configuring the ARIMA model with optimal parameters

# %%
model = auto_arima(train_data, start_p=1, start_q=1, max_p=4, max_q=4, seasonal=True, m=12, 
                   d=1, D=1, trace=True, error_action='ignore', suppress_warnings=True)

# %%
print("Model AIC:", model.aic())

# %% [markdown]
# Fitting the model to training data

# %%
model.fit(train_data)

# %% [markdown]
# Making Forecasts<br>
# Now we will make forecasts based on the fitted model.<br>
# Forecasting for the test period

# %%
forecast_values = model.predict(n_periods=len(test_data))

# %% [markdown]
# Comparing forecasted and actual values

# %%
forecast_values = forecast_values.reset_index(drop=True)
test_data['Forecast'] = forecast_values
print(test_data['Forecast'])

# %% [markdown]
# Plotting actual vs forecasted values

# %%
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['Value'], label='Training Data')  # Use index for train_data
plt.plot(test_data['YYYYMM'], test_data['Value'], label='Actual Data', color='blue')
plt.plot(test_data['YYYYMM'], test_data['Forecast'], label='Forecasted Data', color='red')
plt.title("Natural Gas Emissions: Actual vs Forecasted")
plt.xlabel("Date")
plt.ylabel("Emissions")
plt.legend()
plt.show()

# %% [markdown]
# Calculating mean absolute error for model evaluation

# %%
print(len(test_data['Value']), len(test_data['Forecast']))

# %%
# Calculating mean absolute error for model evaluation
mae_value = mean_absolute_error(test_data['Value'], test_data['Forecast'])
print(f"Mean Absolute Error: {mae_value}")


