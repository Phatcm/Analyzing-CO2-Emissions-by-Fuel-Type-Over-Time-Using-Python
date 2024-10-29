# CO2 Emissions Analysis

## Overview
This project analyzes carbon dioxide (CO2) emissions from various fuel types over time. Using Python and data visualization libraries, the analysis aims to identify trends, seasonal patterns, and forecast future emissions to inform climate policy and energy management strategies.

## Objectives
- Identify primary contributors to CO2 emissions among fuel types.
- Analyze seasonal patterns in emissions over time.
- Forecast future CO2 emissions to understand potential environmental impacts.

## Dataset
The dataset used in this project is `CO2_T12_06.csv`, which contains monthly CO2 emissions data categorized by fuel type. The dataset includes the following columns:
- **YYYYMM**: Year and month of the observation (YYYYMM format)
- **Value**: Amount of CO2 emissions
- **Description**: Fuel type description

## Prerequisites
To run this project, you'll need to have the following installed:
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `pmdarima`, `sklearn`

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn statsmodels pmdarima scikit-learn
```
## Useage:
1. Clone the repository:
```bash
git clone https://github.com/Phatcm/Analyzing-CO2-Emissions-by-Fuel-Type-Over-Time-Using-Python.git
cd Analyzing-CO2-Emissions-by-Fuel-Type-Over-Time-Using-Python
```

2. Make sure CO2_T12_06.csv dataset is in the project directory.
3. Run the python notebook step by step.

## Analysis Steps
The analysis script performs the following steps:

- Data Import: Reads the dataset and performs initial exploration.
- Data Cleaning: Converts relevant columns to numeric and datetime formats.
- CO2 Emissions Visualization: Plots emissions by fuel type and total emissions.
- Seasonal Decomposition: Analyzes seasonal patterns for natural gas emissions.
- Forecasting: Utilizes the ARIMA model to forecast future emissions and evaluates its accuracy.

## Results
The results of the analysis are visualized using plots generated by Matplotlib. Key findings include:

- Major contributions to CO2 emissions from coal and natural gas.
- Distinct seasonal patterns in natural gas emissions.
- Reliable forecasts of future emissions using the ARIMA model.