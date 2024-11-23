# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State, ALL, MATCH
import statsmodels.api as sm
from dash.exceptions import PreventUpdate
import warnings
import json
import uuid
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Initialize the Dash app with a light Bootstrap theme
external_stylesheets = [dbc.themes.FLATLY]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Portfolio Risk Dashboard"

server = app.server

# Define tickers and their corresponding sectors
tickers = [
    'RELIANCE.NS',    # Reliance Industries Ltd
    'INFY.NS',        # Infosys Ltd
    'HDFCBANK.NS',    # HDFC Bank Ltd
    'TATAMOTORS.NS',  # Tata Motors Ltd
    'LT.NS',          # Larsen & Toubro Ltd
    'MARICO.NS',      # Marico Ltd
    'GRANULES.NS',    # Granules India Ltd
    'IRCTC.NS',       # IRCTC Ltd
    'DEEPAKNTR.NS',   # Deepak Nitrite Ltd
    'CROMPTON.NS',    # Crompton Greaves Consumer Electricals Ltd
    'ASIANPAINT.NS',  # Asian Paints Ltd
    'BAJFINANCE.NS',  # Bajaj Finance Ltd
    'BHARTIARTL.NS',  # Bharti Airtel Ltd
    'DRREDDY.NS',     # Dr. Reddy's Laboratories Ltd
    'EICHERMOT.NS',   # Eicher Motors Ltd
    'ITC.NS',         # ITC Ltd
    'KOTAKBANK.NS',   # Kotak Mahindra Bank Ltd
    'M&M.NS',         # Mahindra & Mahindra Ltd
    'NESTLEIND.NS',   # Nestle India Ltd
    'TCS.NS'          # Tata Consultancy Services Ltd
]

# Mapping of tickers to sectors
ticker_sector = {
    'RELIANCE.NS': 'Energy',
    'INFY.NS': 'Technology',
    'HDFCBANK.NS': 'Financials',
    'TATAMOTORS.NS': 'Automobiles',
    'LT.NS': 'Industrials',
    'MARICO.NS': 'Consumer Staples',
    'GRANULES.NS': 'Pharmaceuticals',
    'IRCTC.NS': 'Consumer Discretionary',
    'DEEPAKNTR.NS': 'Chemicals',
    'CROMPTON.NS': 'Consumer Durables',
    'ASIANPAINT.NS': 'Consumer Durables',
    'BAJFINANCE.NS': 'Financials',
    'BHARTIARTL.NS': 'Telecommunications',
    'DRREDDY.NS': 'Healthcare',
    'EICHERMOT.NS': 'Automobiles',
    'ITC.NS': 'Consumer Staples',
    'KOTAKBANK.NS': 'Financials',
    'M&M.NS': 'Automobiles',
    'NESTLEIND.NS': 'Consumer Staples',
    'TCS.NS': 'Technology'
}

# Create a custom session with updated headers
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})

# Retrieve historical price data for the past 5 years using the custom session
data = yf.download(tickers, period='5y', session=session)['Adj Close']

# Drop any columns with all NaN (in case any stock didn't have data)
data.dropna(axis=1, how='all', inplace=True)

# Forward-fill and backward-fill missing data
data.ffill(inplace=True)
data.bfill(inplace=True)

# Calculate daily returns
returns = data.pct_change().dropna()

# Define function to calculate PCA weights
def calculate_pca_weights(returns):
    pca = PCA()
    pca.fit(returns)

    # Extract loadings
    loadings = pca.components_[0]

    # Assign weights inversely proportional to the absolute value of the first principal component
    weights = 1 / np.abs(loadings)
    weights /= weights.sum()
    return weights

# Define function to calculate Risk Parity weights
def calculate_risk_parity_weights(returns):
    # Estimate the covariance matrix
    cov_matrix = returns.cov()

    # Number of assets
    n = len(returns.columns)

    # Initial weights
    init_weights = np.ones(n) / n

    # Objective function to minimize
    def objective(weights):
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib
        risk_contrib_percent = risk_contrib / portfolio_variance
        # We want all risk contributions to be equal
        target = np.ones(len(weights)) / len(weights)
        return np.sum((risk_contrib_percent - target) ** 2)

    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    # Bounds: weights between 0 and 1
    bounds = tuple((0, 1) for _ in range(n))

    result = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        return result.x
    else:
        # Optimization failed
        return None  # Will handle in the callback

# Define function to calculate Black-Litterman weights
def calculate_black_litterman_weights(returns, tau, P, Q, omega, risk_free_rate=0.05):
    # Covariance matrix
    cov_matrix = returns.cov()
    market_weights = np.ones(len(returns.columns)) / len(returns.columns)
    delta = 2.5  # Market risk aversion coefficient

    # Equilibrium excess returns (Pi)
    pi = delta * np.dot(cov_matrix, market_weights)

    # Convert P, Q, omega to numpy arrays
    P = np.array(P)
    Q = np.array(Q)
    omega = np.array(omega)

    # Black-Litterman posterior expected returns (mu_bl)
    try:
        inv_tau_cov = np.linalg.inv(tau * cov_matrix)
        inv_omega = np.linalg.inv(omega)
        middle = inv_tau_cov + P.T @ inv_omega @ P
        mu_bl = np.linalg.inv(middle) @ (inv_tau_cov @ pi + P.T @ inv_omega @ Q)
    except np.linalg.LinAlgError:
        # Handle inversion error by returning None
        return None

    # Mean-Variance Optimization using mu_bl
    # Objective: Maximize Sharpe Ratio
    expected_returns = mu_bl
    cov_matrix = returns.cov()

    def sharpe_ratio_neg(weights):
        portfolio_return = np.dot(weights, expected_returns) * 252  # Annualize
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_std  # Negative for minimization

    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    # Bounds: weights between 0 and 1
    n = len(returns.columns)
    bounds = tuple((0, 1) for _ in range(n))
    # Initial guess
    init_weights = np.ones(n) / n

    result = minimize(sharpe_ratio_neg, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        return result.x
    else:
        # Optimization failed
        return None  # Will handle in the callback

# Define function to calculate Maximum Sharpe Ratio weights
def calculate_max_sharpe_weights(returns, risk_free_rate=0.05):
    cov_matrix = returns.cov()
    expected_returns = returns.mean() * 252  # Annualized expected returns
    n = len(returns.columns)

    # Objective function: maximize Sharpe Ratio
    def objective(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        return -sharpe_ratio  # Negative because we minimize in optimization

    # Constraints
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
    )

    # Bounds: weights between 0 and 1
    bounds = tuple((0, 1) for _ in range(n))

    # Initial guess
    init_weights = np.ones(n) / n

    result = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        return result.x
    else:
        # Optimization failed
        return None  # Will handle in the callback

# Define function to calculate Equal Weights
def calculate_equal_weights(returns):
    n = len(returns.columns)
    return np.ones(n) / n

# Define function to calculate Minimum Variance weights
def calculate_min_variance_weights(returns):
    cov_matrix = returns.cov()

    n = len(returns.columns)
    init_weights = np.ones(n) / n

    # Objective: minimize portfolio variance
    def objective(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    # Bounds: weights between 0 and 1
    bounds = tuple((0, 1) for _ in range(n))

    result = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        return result.x
    else:
        # Optimization failed
        return None  # Will handle in the callback

# Define function to calculate VaR and ES at a single confidence level
def calculate_var_es_single(returns, weights, confidence_level=0.95, time_horizon=1, method='historical'):
    # Calculate the portfolio returns
    portfolio_returns = returns.dot(weights)

    # Adjust returns for time horizon
    portfolio_returns = portfolio_returns * np.sqrt(time_horizon)

    if method == 'historical':
        # Historical VaR and ES
        sorted_returns = np.sort(portfolio_returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        var = -sorted_returns[index]
        es = -sorted_returns[:index].mean()
    elif method == 'parametric':
        # Parametric VaR and ES assuming normal distribution
        mean = portfolio_returns.mean()
        std = portfolio_returns.std()
        var = - (mean + std * stats.norm.ppf(1 - confidence_level))
        es = - (mean + std * stats.norm.pdf(stats.norm.ppf(1 - confidence_level)) / (1 - confidence_level))
    else:
        raise ValueError("Invalid method specified. Use 'historical' or 'parametric'.")

    return var, es, portfolio_returns

# Function to calculate additional risk metrics
def calculate_risk_metrics(returns, weights, portfolio_returns, time_horizon):
    # Annualized returns and volatility
    annualized_return = np.dot(weights, returns.mean()) * 252
    annualized_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

    # Sharpe Ratio
    risk_free_rate = 0.05
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    # Sortino Ratio
    negative_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = negative_returns.std() * np.sqrt(252)
    sortino_ratio = (annualized_return - risk_free_rate) / downside_std

    # Maximum Drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Calmar Ratio
    calmar_ratio = annualized_return / abs(max_drawdown)

    # Beta and Alpha (using NIFTY 50 as benchmark)
    nifty_data = yf.download('^NSEI', period='5y', session=session)['Adj Close'].pct_change().dropna()

    # Remove timezone information from indices
    nifty_data.index = nifty_data.index.tz_localize(None)
    portfolio_returns.index = portfolio_returns.index.tz_localize(None)

    # Align indices
    nifty_returns = nifty_data.reindex(portfolio_returns.index)

    # Handle missing data
    nifty_returns.fillna(method='ffill', inplace=True)
    nifty_returns.fillna(method='bfill', inplace=True)

    benchmark_returns = nifty_returns

    X = sm.add_constant(benchmark_returns)
    model = sm.OLS(portfolio_returns, X).fit()
    alpha = model.params['const'] * 252  # Annualized
    beta = model.params[benchmark_returns.name]

    # Tracking Error
    tracking_error = np.std(portfolio_returns - benchmark_returns) * np.sqrt(252)

    # Information Ratio
    information_ratio = (annualized_return - benchmark_returns.mean() * 252) / tracking_error

    return {
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar_ratio,
        'Alpha': alpha,
        'Beta': beta,
        'Tracking Error': tracking_error,
        'Information Ratio': information_ratio
    }

# Function to perform rolling VaR and ES calculations for trend analysis
def calculate_rolling_var_es(returns, weights, window=252, confidence_level=0.95, method='historical'):
    portfolio_returns = returns.dot(weights)
    portfolio_returns = portfolio_returns * np.sqrt(1)  # Assuming daily returns; adjust if different

    rolling_var = portfolio_returns.rolling(window).apply(
        lambda x: -np.percentile(x, (1 - confidence_level) * 100) if method == 'historical' else -(
            x.mean() + x.std() * stats.norm.ppf(1 - confidence_level)
        ),
        raw=True
    )
    rolling_es = portfolio_returns.rolling(window).apply(
        lambda x: -x[x <= -np.percentile(x, (1 - confidence_level) * 100)].mean() if method == 'historical' else -(
            x.mean() + x.std() * stats.norm.pdf(stats.norm.ppf(1 - confidence_level)) / (1 - confidence_level)
        ),
        raw=True
    )
    return rolling_var, rolling_es

# Function to perform backtesting of VaR
def backtest_var(portfolio_returns, var_series):
    violations = portfolio_returns < -var_series
    violation_count = violations.sum()
    total = violations.count()
    violation_rate = violation_count / total
    return violation_count, violation_rate

# Function to perform stress testing
def stress_test(weights, stress_scenarios, returns):
    stress_results = {}
    for scenario_id, details in stress_scenarios.items():
        shocks = details['shocks']
        probability = details['probability']

        # Apply shock to the portfolio
        stressed_weights = weights.copy()
        stressed_weights += shocked_weights(shocks, ticker_sector)
        stressed_weights = np.clip(stressed_weights, 0, 1)
        if stressed_weights.sum() == 0:
            stressed_weights = np.ones(len(weights)) / len(weights)
        else:
            stressed_weights /= stressed_weights.sum()
        stressed_return = np.dot(stressed_weights, returns.mean()) * 252
        stress_results[details['name']] = {'return': stressed_return, 'probability': probability}
    return stress_results

# Function to generate stressed weights based on scenario
def shocked_weights(shock, ticker_sector):
    # Apply shocks to specific sectors
    sector_shocks = {
        'Energy': shock.get('Energy', 0),
        'Technology': shock.get('Technology', 0),
        'Financials': shock.get('Financials', 0),
        'Automobiles': shock.get('Automobiles', 0),
        'Industrials': shock.get('Industrials', 0),
        'Consumer Staples': shock.get('Consumer Staples', 0),
        'Pharmaceuticals': shock.get('Pharmaceuticals', 0),
        'Consumer Discretionary': shock.get('Consumer Discretionary', 0),
        'Chemicals': shock.get('Chemicals', 0),
        'Consumer Durables': shock.get('Consumer Durables', 0),
        'Telecommunications': shock.get('Telecommunications', 0),
        'Healthcare': shock.get('Healthcare', 0)
    }
    shocks = np.array([sector_shocks.get(ticker_sector[ticker], 0) for ticker in tickers])
    return shocks

# Function to perform Monte Carlo simulations
def monte_carlo_simulation(returns, weights, num_simulations=1000, time_horizon=252, distribution='normal', conditional_shocks=None):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    portfolio_mean = np.dot(weights, mean_returns) * (time_horizon / 252)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) * (time_horizon / 252))

    if distribution == 'normal':
        simulated_returns = np.random.normal(portfolio_mean, portfolio_std, num_simulations)
    elif distribution == 't':
        # Student's t-distribution with 3 degrees of freedom
        simulated_returns = np.random.standard_t(df=3, size=num_simulations) * portfolio_std + portfolio_mean
    elif distribution == 'historical':
        # Resample historical returns
        historical_returns = returns.dot(weights) * (time_horizon / 252)
        simulated_returns = np.random.choice(historical_returns, size=num_simulations, replace=True)
    else:
        raise ValueError("Unsupported distribution type.")

    # Apply conditional shocks if any
    if conditional_shocks:
        for shock in conditional_shocks:
            indices = np.random.choice(range(num_simulations), size=shock['count'], replace=False)
            simulated_returns[indices] += shock['impact']

    return simulated_returns

# Function to perform path-based Monte Carlo simulations
def monte_carlo_simulation_paths(returns, weights, num_simulations=1000, time_horizon=252, distribution='normal', conditional_shocks=None):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    portfolio_mean_daily = np.dot(weights, mean_returns)
    portfolio_cov_daily = np.dot(weights.T, np.dot(cov_matrix, weights))

    if distribution == 'normal':
        simulated_returns = np.random.normal(portfolio_mean_daily, np.sqrt(portfolio_cov_daily), (num_simulations, time_horizon))
    elif distribution == 't':
        simulated_returns = np.random.standard_t(df=3, size=(num_simulations, time_horizon)) * np.sqrt(portfolio_cov_daily) + portfolio_mean_daily
    elif distribution == 'historical':
        historical_returns = returns.dot(weights)
        simulated_returns = np.random.choice(historical_returns, size=(num_simulations, time_horizon), replace=True)
    else:
        raise ValueError("Unsupported distribution type.")

    # Apply conditional shocks if any (e.g., apply to specific paths)
    if conditional_shocks:
        for shock in conditional_shocks:
            path_indices = np.random.choice(range(num_simulations), size=shock['count'], replace=False)
            day_indices = np.random.choice(range(time_horizon), size=shock['count'], replace=True)
            for p, d in zip(path_indices, day_indices):
                simulated_returns[p, d] += shock['impact']

    # Calculate cumulative returns
    cumulative_returns = (1 + simulated_returns).cumprod(axis=1) * 100000  # Assuming portfolio value ₹100,000
    return cumulative_returns

# Define weighting methods
weight_methods = {
    'PCA': calculate_pca_weights,
    'RiskParity': calculate_risk_parity_weights,
    'BlackLitterman': calculate_black_litterman_weights,
    'MaxSharpe': calculate_max_sharpe_weights,
    'Equal': calculate_equal_weights,
    'MinVariance': calculate_min_variance_weights
}

# Initial stress scenarios with unique UUIDs
initial_stress_scenarios = {
    str(uuid.uuid4()): {'name': 'Market Crash', 'shocks': {'Energy': -0.2, 'Financials': -0.15}, 'probability': 0.3},
    str(uuid.uuid4()): {'name': 'Sector Downturn', 'shocks': {'Technology': -0.25}, 'probability': 0.2},
    str(uuid.uuid4()): {'name': 'Interest Rate Spike', 'shocks': {'Financials': -0.1}, 'probability': 0.15},
    str(uuid.uuid4()): {'name': 'Commodity Price Drop', 'shocks': {'Energy': -0.3}, 'probability': 0.2},
    str(uuid.uuid4()): {'name': 'Regulatory Shock', 'shocks': {'Healthcare': -0.2}, 'probability': 0.15}
}

# Layout of the Dash app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Advanced Portfolio Risk Dashboard"), width={'size': 12}, style={'textAlign': 'center', 'marginTop': 30})
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Select Time Interval:"),
            dcc.Dropdown(
                id='time-horizon',
                options=[
                    {'label': '1 Day', 'value': 1},
                    {'label': '1 Week', 'value': 5},
                    {'label': '1 Month', 'value': 21},
                    {'label': '3 Months', 'value': 63},
                    {'label': '6 Months', 'value': 126},
                    {'label': '1 Year', 'value': 252}
                ],
                value=1
            ),
            html.Br(),
            html.Label("Select Weighting Method:"),
            dcc.Dropdown(
                id='weight-method',
                options=[
                    {'label': 'PCA Weights', 'value': 'PCA'},
                    {'label': 'Risk Parity', 'value': 'RiskParity'},
                    {'label': 'Black-Litterman', 'value': 'BlackLitterman'},
                    {'label': 'Maximum Sharpe Ratio', 'value': 'MaxSharpe'},
                    {'label': 'Equal Weight', 'value': 'Equal'},
                    {'label': 'Minimum Variance', 'value': 'MinVariance'}
                ],
                value='PCA'
            ),
            html.Br(),
            html.Div(id='black-litterman-inputs', children=[]),
            html.Br(),
            html.Label("Select VaR Method:"),
            dcc.RadioItems(
                id='var-method',
                options=[
                    {'label': 'Historical Simulation', 'value': 'historical'},
                    {'label': 'Parametric (Normal Distribution)', 'value': 'parametric'}
                ],
                value='historical',
                labelStyle={'display': 'block', 'margin-bottom': '10px'}
            ),
            html.Br(),
            html.Label("Select Confidence Level (%):"),
            dcc.Slider(
                id='confidence-level',
                min=90,
                max=99,
                step=0.5,
                value=95,
                marks={i: f'{i}%' for i in range(90, 100)}
            ),
            html.Br(),
            html.Label("Enter Portfolio Value (₹):"),
            dcc.Input(
                id='portfolio-value',
                type='number',
                value=100000,
                min=0,
                step=1,
                style={'width': '100%'}
            ),
            html.Br(),
            html.Br(),
            html.H5("Monte Carlo Simulation Controls"),
            html.Label("Number of Simulations:"),
            dcc.Input(
                id='mc-num-simulations',
                type='number',
                value=1000,
                min=100,
                step=100,
                style={'width': '100%'}
            ),
            html.Br(),
            html.Label("Time Horizon (Trading Days):"),
            dcc.Input(
                id='mc-time-horizon',
                type='number',
                value=252,
                min=1,
                step=1,
                style={'width': '100%'}
            ),
            html.Br(),
            html.Label("Select Distribution for Simulations:"),
            dcc.Dropdown(
                id='mc-distribution',
                options=[
                    {'label': 'Normal Distribution', 'value': 'normal'},
                    {'label': "Student's t-Distribution", 'value': 't'},
                    {'label': 'Historical Distribution', 'value': 'historical'}
                ],
                value='normal'
            ),
            html.Br(),
            html.Label("Select Simulation Type:"),
            dcc.RadioItems(
                id='mc-simulation-type',
                options=[
                    {'label': 'End-of-Period Returns', 'value': 'returns'},
                    {'label': 'Path Simulations', 'value': 'paths'}
                ],
                value='returns',
                labelStyle={'display': 'block', 'margin-bottom': '10px'}
            ),
            html.Br(),
            html.Label("Set Return Threshold for Alerts (₹):"),
            dcc.Input(
                id='mc-return-threshold',
                type='number',
                value=-20000,
                step=1000,
                style={'width': '100%'}
            ),
            html.Br(),
            html.H5("Dynamic Stress Scenarios"),
            html.Div([
                dbc.Button("Add Scenario", id="add-scenario-button", color="secondary", size="sm", className="mb-2"),
                html.Div(id='stress-scenarios-container', children=[
                    # Initial stress scenarios will be loaded here
                ]),
                dbc.Button("Save Scenarios", id="save-scenarios-button", color="primary", size="sm", className="mt-2")
            ]),
            html.Br(),
            dbc.Button('Update', id='update-button', n_clicks=0, color='primary'),
            # Hidden divs to store data
            dcc.Store(id='bl-params', data={'tau': 0.05, 'P': [[1] + [0]*(len(tickers)-1)], 'Q': [0.02], 'omega': [[0.0001]]}),
            dcc.Store(id='stress-scenarios-store', data=initial_stress_scenarios)
        ], width=4, style={'padding': '20px', 'overflowY': 'auto', 'maxHeight': '90vh'}),
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label='Risk Dashboard', tab_id='tab-risk'),
                dbc.Tab(label='Portfolio Weights', tab_id='tab-weights'),
                dbc.Tab(label='Return Distribution', tab_id='tab-returns'),
                dbc.Tab(label='Risk Contributions', tab_id='tab-contributions'),
                dbc.Tab(label='Risk Metrics Table', tab_id='tab-metrics'),
                dbc.Tab(label='VaR & ES Trend', tab_id='tab-var-es-trend'),
                dbc.Tab(label='Tail Risk & Stress Testing', tab_id='tab-tail-stress'),
                dbc.Tab(label='Monte Carlo Simulation', tab_id='tab-monte-carlo'),
                dbc.Tab(label='Stress Scenarios Heatmap', tab_id='tab-stress-heatmap')
            ], id='tabs', active_tab='tab-risk', style={'marginBottom': '20px'}),
            html.Div(id='tab-content')
        ], width=8, style={'padding': '20px'}),
    ]),
    # Watermark at the bottom right
    html.Div([
        html.A(
            "Made by: Nirbhai, BITS Pilani",
            href="https://www.linkedin.com/in/nirbhai10/",
            target="_blank",
            style={
                'position': 'fixed',
                'right': '10px',
                'bottom': '10px',
                'fontSize': '10px',
                'color': 'gray',
                'textDecoration': 'none'
            }
        )
    ]),
    # Modal for Black-Litterman Parameters
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Black-Litterman Parameters")),
            dbc.ModalBody([
                html.Label("Tau (Scaling Factor):"),
                dbc.Input(id='bl-tau', type='number', value=0.05, step=0.01),
                html.Br(),
                html.Label("Number of Views:"),
                dbc.Input(id='bl-num-views', type='number', value=1, min=1, step=1),
                html.Br(),
                html.Div(id='bl-views-inputs')
            ]),
            dbc.ModalFooter([
                dbc.Button("Save", id="save-bl-params", className="ms-auto", n_clicks=0),
                dbc.Button("Close", id="close-bl-modal", className="ms-2", n_clicks=0)
            ]),
        ],
        id="bl-modal",
        is_open=False,
        size='lg'
    )
], fluid=True)  # fluid=True is correctly applied to dbc.Container

# Callback to display Black-Litterman parameter inputs when selected
@app.callback(
    Output('black-litterman-inputs', 'children'),
    [Input('weight-method', 'value')]
)
def display_bl_inputs(weight_method):
    if weight_method == 'BlackLitterman':
        return html.Div([
            html.H5("Black-Litterman Parameters"),
            html.Br(),
            html.Label("Tau (Scaling Factor):"),
            dbc.Input(id='bl-tau', type='number', value=0.05, step=0.01),
            html.Br(),
            html.Label("Number of Views:"),
            dbc.Input(id='bl-num-views', type='number', value=1, min=1, step=1),
            html.Br(),
            html.Div(id='bl-views-inputs')
        ])
    return no_update

# Callback to dynamically generate views based on the number of views
@app.callback(
    Output('bl-views-inputs', 'children'),
    [Input('bl-num-views', 'value')]
)
def update_bl_views(num_views):
    if num_views < 1:
        return no_update
    inputs = []
    for i in range(num_views):
        inputs.append(
            dbc.Card([
                dbc.CardHeader(f"View {i+1}"),
                dbc.CardBody([
                    html.Label(f"Select Asset for View {i+1}:"),
                    dcc.Dropdown(
                        id={'type': 'bl-view-asset', 'index': i},
                        options=[{'label': ticker, 'value': ticker} for ticker in tickers],
                        value=tickers[i % len(tickers)]
                    ),
                    html.Br(),
                    html.Label(f"Expected Excess Return (Q) for View {i+1}:"),
                    dbc.Input(id={'type': 'bl-view-q', 'index': i}, type='number', value=0.02, step=0.01),
                    html.Br(),
                    html.Label(f"Uncertainty (Omega) for View {i+1}:"),
                    dbc.Input(id={'type': 'bl-view-omega', 'index': i}, type='number', value=0.0001, step=0.0001),
                ])
            ], style={'margin-bottom': '20px'})
        )
    return inputs

# Combined callback to manage adding and removing stress scenarios
@app.callback(
    Output('stress-scenarios-container', 'children'),
    [
        Input('add-scenario-button', 'n_clicks'),
        Input({'type': 'remove-scenario', 'index': ALL}, 'n_clicks')
    ],
    [
        State('stress-scenarios-container', 'children'),
        State('stress-scenarios-store', 'data')
    ]
)
def manage_stress_scenarios(add_n_clicks, remove_n_clicks, children, stress_scenarios_data):
    ctx = dash.callback_context

    if not ctx.triggered:
        # Initial load: Populate scenarios from the store
        return [
            dbc.Card([
                dbc.CardHeader(f"Scenario {i + 1}: {details['name']}"),
                dbc.CardBody([
                    html.Label("Scenario Name:"),
                    dbc.Input(id={'type': 'scenario-name', 'index': scenario_id}, type='text', value=details['name'], disabled=True),
                    html.Br(),
                    html.Label("Select Affected Sectors:"),
                    dcc.Dropdown(
                        id={'type': 'scenario-sectors', 'index': scenario_id},
                        options=[{'label': sector, 'value': sector} for sector in set(ticker_sector.values())],
                        multi=True,
                        value=list(details['shocks'].keys()),
                        disabled=True
                    ),
                    html.Br(),
                    html.Label("Specify Shock Magnitudes (in %):"),
                    dbc.Input(
                        id={'type': 'scenario-shock', 'index': scenario_id},
                        type='text',
                        value=','.join([f"{sector}:{magnitude}" for sector, magnitude in details['shocks'].items()]),
                        placeholder='Sector1:-0.2,Sector2:-0.1',
                        disabled=True
                    ),
                    html.Br(),
                    html.Label("Assign Probability (0-1):"),
                    dbc.Input(
                        id={'type': 'scenario-probability', 'index': scenario_id},
                        type='number',
                        value=details['probability'],
                        min=0,
                        max=1,
                        step=0.01,
                        disabled=True
                    ),
                    html.Br(),
                    dbc.Button("Remove Scenario", id={'type': 'remove-scenario', 'index': scenario_id}, color="danger", size="sm")
                ])
            ], style={'margin-bottom': '20px'})
            for i, (scenario_id, details) in enumerate(stress_scenarios_data.items())
        ]

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'add-scenario-button':
        # Adding a new scenario
        new_id = str(uuid.uuid4())  # Unique identifier
        new_scenario = dbc.Card([
            dbc.CardHeader(f"New Scenario"),
            dbc.CardBody([
                html.Label("Scenario Name:"),
                dbc.Input(id={'type': 'scenario-name', 'index': new_id}, type='text', placeholder="Enter scenario name"),
                html.Br(),
                html.Label("Select Affected Sectors:"),
                dcc.Dropdown(
                    id={'type': 'scenario-sectors', 'index': new_id},
                    options=[{'label': sector, 'value': sector} for sector in set(ticker_sector.values())],
                    multi=True
                ),
                html.Br(),
                html.Label("Specify Shock Magnitudes (in %):"),
                dbc.Input(
                    id={'type': 'scenario-shock', 'index': new_id},
                    type='text',
                    placeholder='Sector1:-0.2,Sector2:-0.1'
                ),
                html.Br(),
                html.Label("Assign Probability (0-1):"),
                dbc.Input(
                    id={'type': 'scenario-probability', 'index': new_id},
                    type='number',
                    placeholder='0.2',
                    min=0,
                    max=1,
                    step=0.01
                ),
                html.Br(),
                dbc.Button("Remove Scenario", id={'type': 'remove-scenario', 'index': new_id}, color="danger", size="sm")
            ])
        ], style={'margin-bottom': '20px'})
        children.append(new_scenario)

    elif any(triggered_id.startswith('{"type": "remove-scenario"')):
        # Removing a scenario
        try:
            # Extract the index of the scenario to remove
            triggered_dict = json.loads(triggered_id.replace("'", '"'))
            scenario_id_to_remove = triggered_dict['index']
            # Remove the scenario with the matching UUID
            children = [
                child for child in children
                if child['props']['children'][0]['props']['children'][0]['props']['children'] != f"Scenario {scenario_id_to_remove}: {stress_scenarios_data[scenario_id_to_remove]['name']}"
            ]
            # Remove the scenario from the store data
            stress_scenarios_data.pop(scenario_id_to_remove, None)
        except Exception as e:
            print(f"Error removing scenario: {e}")

    return children

# Callback to save stress scenarios
@app.callback(
    Output('stress-scenarios-store', 'data'),
    [Input('save-scenarios-button', 'n_clicks')],
    [
        State({'type': 'scenario-name', 'index': ALL}, 'value'),
        State({'type': 'scenario-sectors', 'index': ALL}, 'value'),
        State({'type': 'scenario-shock', 'index': ALL}, 'value'),
        State({'type': 'scenario-probability', 'index': ALL}, 'value'),
        State({'type': 'scenario-name', 'index': ALL}, 'id'),
        State('stress-scenarios-store', 'data')
    ],
    prevent_initial_call=True
)
def save_stress_scenarios(n_clicks, names, sectors, shocks, probabilities, ids, existing_data):
    if n_clicks is None:
        raise PreventUpdate

    new_scenarios = {}
    total_prob = 0
    for i in range(len(names)):
        name = names[i]
        affected_sectors = sectors[i] if sectors[i] else []
        shock_input = shocks[i]
        probability = probabilities[i] if i < len(probabilities) else 0.0
        scenario_id = ids[i]['index']
        
        try:
            shock_dict = {}
            if shock_input:
                for pair in shock_input.split(','):
                    if ':' in pair:
                        sector, magnitude = pair.split(':')
                        shock_dict[sector.strip()] = float(magnitude.strip())
            
            if name:  # Ensure scenario has a name
                new_scenarios[scenario_id] = {
                    'name': name,
                    'shocks': shock_dict,
                    'probability': probability
                }
                total_prob += probability
        except Exception as e:
            print(f"Error processing scenario {i+1}: {e}")
            continue  # Skip invalid entries

    # Normalize probabilities if they don't sum to 1
    if total_prob != 1 and total_prob != 0:
        for scenario in new_scenarios:
            new_scenarios[scenario]['probability'] /= total_prob

    return new_scenarios

# Callback to save Black-Litterman parameters
@app.callback(
    Output('bl-params', 'data'),
    [Input('save-bl-params', 'n_clicks')],
    [
        State('bl-tau', 'value'),
        State('bl-num-views', 'value'),
        State({'type': 'bl-view-asset', 'index': ALL}, 'value'),
        State({'type': 'bl-view-q', 'index': ALL}, 'value'),
        State({'type': 'bl-view-omega', 'index': ALL}, 'value')
    ],
    prevent_initial_call=True
)
def save_bl_params(n_clicks, tau, num_views, assets, Q, omega):
    if n_clicks > 0:
        P = []
        for asset in assets:
            p = [0] * len(tickers)
            if asset in tickers:
                p[tickers.index(asset)] = 1
            P.append(p)
        Q = Q[:num_views]
        # Correctly construct Omega as a diagonal matrix
        omega_diag = np.diag([o for o in omega[:num_views]]).tolist()
        return {'tau': tau, 'P': P, 'Q': Q, 'omega': omega_diag}
    return no_update

# Callback to handle opening and closing the modal
@app.callback(
    Output("bl-modal", "is_open"),
    [
        Input("weight-method", "value"),
        Input("close-bl-modal", "n_clicks"),
        Input("save-bl-params", "n_clicks")
    ],
    [State("bl-modal", "is_open")]
)
def manage_bl_modal(weight_method, close_n_clicks, save_n_clicks, is_open):
    triggered_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == "weight-method":
        if weight_method == "BlackLitterman" and not is_open:
            return True
        else:
            return False
    elif triggered_id in ["close-bl-modal", "save-bl-params"]:
        return False
    return is_open

# Callback for Risk Dashboard Tab
@app.callback(
    Output('tab-content', 'children'),
    [
        Input('update-button', 'n_clicks'),
        Input('tabs', 'active_tab')
    ],
    [
        State('time-horizon', 'value'),
        State('weight-method', 'value'),
        State('var-method', 'value'),
        State('confidence-level', 'value'),
        State('portfolio-value', 'value'),
        State('bl-params', 'data'),
        State('mc-num-simulations', 'value'),
        State('mc-time-horizon', 'value'),
        State('mc-distribution', 'value'),
        State('mc-simulation-type', 'value'),
        State('mc-return-threshold', 'value'),
        State('stress-scenarios-store', 'data')
    ]
)
def render_content(n_clicks, active_tab, time_horizon, weight_method, var_method, confidence_level, portfolio_value, bl_params,
                  mc_num_simulations, mc_time_horizon, mc_distribution, mc_simulation_type, mc_return_threshold, stress_scenarios):
    if n_clicks == 0:
        raise PreventUpdate

    # Update weights based on the selected method
    weights = np.ones(len(returns.columns)) / len(returns.columns)  # Default equal weights
    optimization_error = False
    optimization_message = ""

    if weight_method in weight_methods:
        if weight_method == 'BlackLitterman':
            weights_result = weight_methods[weight_method](returns, bl_params['tau'], bl_params['P'], bl_params['Q'], bl_params['omega'])
            if weights_result is not None:
                weights = weights_result
            else:
                optimization_error = True
                optimization_message = "Black-Litterman optimization failed. Using equal weights."
        else:
            weights_result = weight_methods[weight_method](returns)
            if weights_result is not None:
                weights = weights_result
            else:
                optimization_error = True
                optimization_message = f"{weight_method} optimization did not converge. Using equal weights."

    # Convert confidence level to decimal
    confidence_level_decimal = confidence_level / 100.0

    # Calculate VaR and ES at the selected confidence level
    try:
        var_percent, es_percent, portfolio_returns = calculate_var_es_single(
            returns, weights, confidence_level_decimal, time_horizon, method=var_method
        )
    except Exception as e:
        return html.Div([
            dbc.Alert(f"Error in calculating VaR and ES: {str(e)}", color="danger")
        ])

    # Calculate VaR and ES in monetary terms
    var_monetary = var_percent * portfolio_value
    es_monetary = es_percent * portfolio_value

    # Calculate additional risk metrics
    try:
        risk_metrics = calculate_risk_metrics(returns, weights, portfolio_returns, time_horizon)
    except Exception as e:
        return html.Div([
            dbc.Alert(f"Error in calculating risk metrics: {str(e)}", color="danger")
        ])

    # Handle optimization errors by displaying a warning
    if optimization_error:
        optimization_alert = dbc.Alert(optimization_message, color="warning")
    else:
        optimization_alert = None  # Do not include any alert

    if active_tab == 'tab-risk':
        # Create cards to display VaR and ES
        var_card = dbc.Card(
            [
                dbc.CardHeader(f"Value at Risk (VaR) at {int(confidence_level_decimal * 100)}% Confidence Level"),
                dbc.CardBody(
                    [
                        html.H5(f"{var_percent:.2%}", className="card-title"),
                        html.P(f"Monetary Value: ₹{var_monetary:,.2f}", className="card-text"),
                    ]
                ),
            ],
            color="danger",
            inverse=True,
        )

        es_card = dbc.Card(
            [
                dbc.CardHeader(f"Expected Shortfall (ES) at {int(confidence_level_decimal * 100)}% Confidence Level"),
                dbc.CardBody(
                    [
                        html.H5(f"{es_percent:.2%}", className="card-title"),
                        html.P(f"Monetary Value: ₹{es_monetary:,.2f}", className="card-text"),
                    ]
                ),
            ],
            color="warning",
            inverse=True,
        )

        # Layout the cards side by side
        cards = dbc.Row(
            [
                dbc.Col(var_card, width=6),
                dbc.Col(es_card, width=6),
            ],
            className="mb-4",
        )

        # Create an area chart for the CDF
        sorted_returns = np.sort(portfolio_returns)
        cumulative_probs = np.linspace(0, 1, len(sorted_returns))

        risk_fig = go.Figure()

        # Area under curve
        risk_fig.add_trace(go.Scatter(
            x=sorted_returns,
            y=cumulative_probs,
            mode='lines',
            name='CDF of Portfolio Returns',
            line=dict(color='#17BECF'),
            fill='tozeroy',
        ))

        risk_fig.update_layout(
            title='Portfolio Return Cumulative Distribution Function (CDF)',
            xaxis_title='Portfolio Returns (₹)',
            yaxis_title='Cumulative Probability',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=14),
            legend=dict(font_size=12),
            title_font_size=16,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        children = []
        if optimization_alert:
            children.append(optimization_alert)

        children.extend([
            cards,
            dcc.Graph(figure=risk_fig)
        ])

        return html.Div(children)

    elif active_tab == 'tab-weights':
        # Create the weights bar chart
        weights_df = pd.DataFrame({
            'Ticker': returns.columns,
            'Weight': weights
        })
        weights_fig = px.bar(
            weights_df,
            x='Ticker',
            y='Weight',
            title='Portfolio Weights',
            color='Weight',
            color_continuous_scale='Blues'
        )
        weights_fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=14),
            xaxis_tickangle=-65,
            title_font_size=16,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        weights_fig.update_xaxes(tickfont_size=12)
        weights_fig.update_yaxes(title_font_size=14)
        weights_fig.update_traces(texttemplate='%{y:.2%}', textposition='outside')

        children = []
        if optimization_alert:
            children.append(optimization_alert)

        children.extend([
            dcc.Graph(figure=weights_fig)
        ])

        return html.Div(children)

    elif active_tab == 'tab-returns':
        # Create the return distribution histogram with KDE and CDF
        hist_data = portfolio_returns

        # Create the histogram figure
        hist_fig = go.Figure()

        # Add histogram
        hist_fig.add_trace(go.Histogram(
            x=hist_data,
            nbinsx=30,
            name='Simulated Returns',
            marker_color='#636EFA',
            opacity=0.75
        ))

        # Add density curve using gaussian_kde from scipy
        kde = gaussian_kde(hist_data)
        x_range = np.linspace(hist_data.min(), hist_data.max(), 1000)
        kde_values = kde(x_range)

        hist_fig.add_trace(go.Scatter(
            x=x_range,
            y=kde_values * len(hist_data) * (x_range[1] - x_range[0]),  # Scale to match histogram
            mode='lines',
            name='KDE',
            line=dict(color='orange', width=2)
        ))

        # Add CDF
        sorted_returns = np.sort(hist_data)
        cumulative_probs = np.linspace(0, 1, len(sorted_returns))

        hist_fig.add_trace(go.Scatter(
            x=sorted_returns,
            y=cumulative_probs,
            mode='lines',
            name='CDF',
            line=dict(color='green', width=2, dash='dash')
        ))

        # Adjust x-axis to better fit the distribution
        x_min = max(hist_data.min(), np.percentile(hist_data, 1))
        x_max = min(hist_data.max(), np.percentile(hist_data, 99))
        hist_fig.update_xaxes(range=[x_min, x_max])

        hist_fig.update_layout(
            title='Portfolio Return Distribution',
            xaxis_title='Returns (₹)',
            yaxis_title='Frequency',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=14),
            bargap=0.2,
            legend=dict(font_size=12),
            title_font_size=16,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        hist_fig.update_xaxes(title_font_size=14)
        hist_fig.update_yaxes(title_font_size=14)

        # Display VaR and ES values separately below the graph
        var_text = f"Value at Risk (VaR) at {int(confidence_level_decimal * 100)}% confidence level: ₹{var_monetary:,.2f}"
        es_text = f"Expected Shortfall (ES) at {int(confidence_level_decimal * 100)}% confidence level: ₹{es_monetary:,.2f}"

        children = []
        if optimization_alert:
            children.append(optimization_alert)

        children.extend([
            html.Div([
                html.P(var_text, style={'fontSize': 16}),
                html.P(es_text, style={'fontSize': 16})
            ], style={'textAlign': 'center', 'marginBottom': 20}),
            dcc.Graph(figure=hist_fig)
        ])

        return html.Div(children)

    elif active_tab == 'tab-contributions':
        # Risk Contributions Treemap
        cov_matrix = returns.cov()
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_std
        risk_contrib = weights * marginal_contrib
        risk_contrib_percent = risk_contrib / portfolio_std

        contrib_df = pd.DataFrame({
            'Ticker': returns.columns,
            'Risk Contribution': risk_contrib_percent
        })

        treemap_fig = px.treemap(
            contrib_df,
            path=['Ticker'],
            values='Risk Contribution',
            title='Risk Contributions Treemap',
            color='Risk Contribution',
            color_continuous_scale='Viridis',
            color_continuous_midpoint=np.average(contrib_df['Risk Contribution'])
        )

        treemap_fig.update_traces(
            texttemplate='%{label}<br>%{value:.2%}',
            textfont_size=14
        )

        treemap_fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=14),
            title_font_size=16,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        children = []
        if optimization_alert:
            children.append(optimization_alert)

        children.extend([
            dcc.Graph(figure=treemap_fig)
        ])

        return html.Div(children)

    elif active_tab == 'tab-metrics':
        # Create a table summarizing risk metrics
        metrics_df = pd.DataFrame.from_dict(risk_metrics, orient='index', columns=['Value'])
        metrics_df.reset_index(inplace=True)
        metrics_df.rename(columns={'index': 'Metric'}, inplace=True)
        metrics_table = dbc.Table.from_dataframe(metrics_df.round(4), striped=True, bordered=True, hover=True)

        children = []
        if optimization_alert:
            children.append(optimization_alert)

        children.extend([
            html.H3('Risk Metrics Summary'),
            html.Div(metrics_table, style={'fontSize': 14})
        ])

        return html.Div(children)

    elif active_tab == 'tab-var-es-trend':
        # Calculate rolling VaR and ES
        window = 252  # 1 year
        try:
            rolling_var, rolling_es = calculate_rolling_var_es(
                returns, weights, window=window, confidence_level=confidence_level_decimal, method=var_method
            )
        except Exception as e:
            return html.Div([
                dbc.Alert(f"Error in calculating rolling VaR and ES: {str(e)}", color="danger")
            ])

        # Create a figure for VaR and ES trends
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(
            x=rolling_var.index,
            y=rolling_var,
            mode='lines',
            name='VaR',
            line=dict(color='red')
        ))
        trend_fig.add_trace(go.Scatter(
            x=rolling_es.index,
            y=rolling_es,
            mode='lines',
            name='ES',
            line=dict(color='orange')
        ))

        trend_fig.update_layout(
            title='Rolling VaR and ES Trend',
            xaxis_title='Date',
            yaxis_title='Risk Value (₹)',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=14),
            legend=dict(font_size=12),
            title_font_size=16,
            margin=dict(l=50, r=50, t=50, b=80)
        )

        # Perform backtesting
        backtest_window = window
        try:
            backtest_var_series, backtest_es_series = calculate_rolling_var_es(
                returns, weights, window=backtest_window, confidence_level=confidence_level_decimal, method=var_method
            )
            violations = returns.dot(weights) < -backtest_var_series
            violation_count = violations.sum()
            violation_rate = violation_count / violations.count()
            backtest_text = f"Backtesting Results: {violation_count} violations out of {violations.count()} observations ({violation_rate:.2%} violation rate)."
        except Exception as e:
            backtest_text = f"Backtesting could not be performed: {str(e)}"

        # Add backtest results below the graph to prevent overlap
        children = []
        if optimization_alert:
            children.append(optimization_alert)

        children.extend([
            dcc.Graph(figure=trend_fig),
            html.Div([
                dbc.Alert(backtest_text, color="info", style={'marginTop': '20px'})
            ], style={'textAlign': 'center'})
        ])

        return html.Div(children)

    elif active_tab == 'tab-tail-stress':
        # Tail Risk Metrics
        portfolio_returns = returns.dot(weights)
        tail_threshold = np.percentile(portfolio_returns, (1 - confidence_level_decimal) * 100)
        tail_returns = portfolio_returns[portfolio_returns <= tail_threshold]

        # Tail Ratio
        mean_return = portfolio_returns.mean()
        tail_mean = tail_returns.mean()
        tail_ratio = mean_return / abs(tail_mean) if abs(tail_mean) != 0 else np.nan

        # Conditional VaR (CVaR)
        cvar = tail_returns.mean()

        tail_metrics = {
            'Tail Ratio': tail_ratio,
            'Conditional VaR (CVaR)': cvar
        }

        tail_metrics_df = pd.DataFrame.from_dict(tail_metrics, orient='index', columns=['Value'])
        tail_metrics_df.reset_index(inplace=True)
        tail_metrics_df.rename(columns={'index': 'Metric'}, inplace=True)

        tail_metrics_table = dbc.Table.from_dataframe(tail_metrics_df.round(4), striped=True, bordered=True, hover=True)

        # Stress Testing
        stress_results = stress_test(weights, stress_scenarios, returns)

        # Calculate expected stressed return based on probabilities
        expected_stressed_return = sum([details['return'] * details['probability'] for details in stress_results.values()])

        stress_df = pd.DataFrame.from_dict(stress_results, orient='index')
        stress_df.reset_index(inplace=True)
        stress_df.rename(columns={'index': 'Scenario'}, inplace=True)
        stress_df['Probability (%)'] = stress_df['probability'] * 100
        stress_df['Expected Return (₹)'] = stress_df['return']

        stress_fig = px.bar(
            stress_df,
            x='Scenario',
            y='Expected Return (₹)',
            title='Stress Testing Results',
            color='Expected Return (₹)',
            color_continuous_scale='Reds',
            labels={'Expected Return (₹)': 'Expected Return (₹)'},
            hover_data=['Probability (%)', 'Expected Return (₹)']
        )
        stress_fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=14),
            title_font_size=16,
            xaxis_tickangle=-45,
            margin=dict(l=50, r=50, t=50, b=100)
        )
        stress_fig.update_yaxes(title_font_size=14)

        children = []
        if optimization_alert:
            children.append(optimization_alert)

        children.extend([
            html.H3('Tail Risk Metrics'),
            tail_metrics_table,
            html.H3('Stress Testing'),
            dcc.Graph(figure=stress_fig)
        ])

        return html.Div(children)

    elif active_tab == 'tab-monte-carlo':
        # Perform Monte Carlo Simulations
        try:
            # Handle simulation based on type
            if mc_simulation_type == 'returns':
                simulations = monte_carlo_simulation(
                    returns, weights, num_simulations=mc_num_simulations, time_horizon=mc_time_horizon, distribution=mc_distribution
                )
            else:
                simulations = monte_carlo_simulation_paths(
                    returns, weights, num_simulations=mc_num_simulations, time_horizon=mc_time_horizon, distribution=mc_distribution
                )
        except Exception as e:
            return html.Div([
                dbc.Alert(f"Error in Monte Carlo Simulation: {str(e)}", color="danger")
            ])

        # Check for simulations below the threshold
        if mc_simulation_type == 'returns':
            exceed_count = np.sum(simulations < (mc_return_threshold / portfolio_value))
        else:
            # For path simulations, check the final portfolio value
            exceed_count = np.sum(simulations[:, -1] < mc_return_threshold)
        exceed_percentage = (exceed_count / mc_num_simulations) * 100

        # Generate alert if threshold is breached
        if exceed_percentage > 0:
            threshold_alert = dbc.Alert(
                f"{exceed_percentage:.2f}% of simulations exceed the return threshold of ₹{mc_return_threshold:,.2f}.",
                color="danger"
            )
        else:
            threshold_alert = None

        # Create simulation plot based on type
        if mc_simulation_type == 'returns':
            # Create histogram with KDE and CDF
            hist_fig = go.Figure()

            # Add histogram
            hist_fig.add_trace(go.Histogram(
                x=simulations,
                nbinsx=50,
                name='Simulated Returns',
                marker_color='#EF553B',
                opacity=0.75
            ))

            # Add density curve using gaussian_kde from scipy
            kde = gaussian_kde(simulations)
            x_range = np.linspace(simulations.min(), simulations.max(), 1000)
            kde_values = kde(x_range)

            hist_fig.add_trace(go.Scatter(
                x=x_range,
                y=kde_values * len(simulations) * (x_range[1] - x_range[0]),  # Scale to match histogram
                mode='lines',
                name='KDE',
                line=dict(color='orange', width=2)
            ))

            # Add CDF
            sorted_returns = np.sort(simulations)
            cumulative_probs = np.linspace(0, 1, len(sorted_returns))

            hist_fig.add_trace(go.Scatter(
                x=sorted_returns,
                y=cumulative_probs,
                mode='lines',
                name='CDF',
                line=dict(color='green', width=2, dash='dash')
            ))

            # Adjust x-axis to better fit the distribution
            x_min = max(simulations.min(), np.percentile(simulations, 1))
            x_max = min(simulations.max(), np.percentile(simulations, 99))
            hist_fig.update_xaxes(range=[x_min, x_max])

            hist_fig.update_layout(
                title='Monte Carlo Simulation of Portfolio Returns',
                xaxis_title='Simulated Returns (₹)',
                yaxis_title='Frequency',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black', size=14),
                bargap=0.2,
                legend=dict(font_size=12),
                title_font_size=16,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            hist_fig.update_xaxes(title_font_size=14)
            hist_fig.update_yaxes(title_font_size=14)

            # Summary statistics
            mc_mean = np.mean(simulations)
            mc_std = np.std(simulations)
            mc_var = -np.percentile(simulations, (1 - confidence_level_decimal) * 100)
            mc_es = -simulations[simulations <= -mc_var].mean()

            # Quantile Metrics
            quantiles = [5, 10, 25, 75, 90, 95]
            quantile_values = np.percentile(simulations, quantiles)

            prob_exceedance = {
                f'Probability of Return below {q}th percentile': f"{np.sum(simulations < quantile_values[i])/mc_num_simulations:.2%}"
                for i, q in enumerate(quantiles)
            }

            # Update summary
            mc_summary = {
                'Simulated Mean Return (₹)': mc_mean,
                'Simulated Std Dev (₹)': mc_std,
                'Monte Carlo VaR (₹)': mc_var,
                'Monte Carlo ES (₹)': mc_es
            }
            mc_summary.update(prob_exceedance)

            mc_summary_df = pd.DataFrame.from_dict(mc_summary, orient='index', columns=['Value'])
            mc_summary_df.reset_index(inplace=True)
            mc_summary_df.rename(columns={'index': 'Metric'}, inplace=True)

            mc_summary_table = dbc.Table.from_dataframe(mc_summary_df.round(4), striped=True, bordered=True, hover=True)

            children = []
            if optimization_alert:
                children.append(optimization_alert)
            if threshold_alert:
                children.append(threshold_alert)

            children.extend([
                dcc.Graph(figure=hist_fig),
                html.H3('Monte Carlo Simulation Summary'),
                html.Div(mc_summary_table, style={'fontSize': 14})
            ])

            return html.Div(children)

        else:
            # Path Simulations Visualization
            path_fig = go.Figure()

            # Limit the number of paths to plot for performance
            subset = min(100, mc_num_simulations)  # Plot up to 100 paths
            for path in simulations[:subset]:
                path_fig.add_trace(go.Scatter(
                    x=np.arange(mc_time_horizon),
                    y=path,
                    mode='lines',
                    line=dict(width=1, color='blue'),
                    opacity=0.1
                ))

            # Add median and confidence intervals
            median_returns = np.median(simulations, axis=0)
            lower_conf = np.percentile(simulations, 2.5, axis=0)
            upper_conf = np.percentile(simulations, 97.5, axis=0)

            path_fig.add_trace(go.Scatter(
                x=np.arange(mc_time_horizon),
                y=median_returns,
                mode='lines',
                name='Median Portfolio Value',
                line=dict(color='red', width=2)
            ))

            path_fig.add_trace(go.Scatter(
                x=np.concatenate([np.arange(mc_time_horizon), mc_time_horizon-1, np.arange(mc_time_horizon-1, -1, -1)]),
                y=np.concatenate([lower_conf, [lower_conf[-1]], upper_conf[::-1]]),
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='95% Confidence Interval'
            ))

            path_fig.update_layout(
                title='Monte Carlo Path Simulations of Portfolio Value',
                xaxis_title='Trading Days',
                yaxis_title='Portfolio Value (₹)',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black', size=14),
                showlegend=True,
                title_font_size=16,
                margin=dict(l=50, r=50, t=50, b=80)
            )
            path_fig.update_xaxes(title_font_size=14)
            path_fig.update_yaxes(title_font_size=14)

            # Summary statistics
            mc_mean = np.mean(simulations)
            mc_std = np.std(simulations)
            mc_var = -np.percentile(simulations, (1 - confidence_level_decimal) * 100)
            mc_es = -simulations[simulations <= -mc_var].mean()

            # Quantile Metrics
            quantiles = [5, 10, 25, 75, 90, 95]
            quantile_values = np.percentile(simulations, quantiles)

            prob_exceedance = {
                f'Probability of Return below {q}th percentile': f"{np.sum(simulations < quantile_values[i])/mc_num_simulations:.2%}"
                for i, q in enumerate(quantiles)
            }

            # Update summary
            mc_summary = {
                'Simulated Mean Return (₹)': mc_mean,
                'Simulated Std Dev (₹)': mc_std,
                'Monte Carlo VaR (₹)': mc_var,
                'Monte Carlo ES (₹)': mc_es
            }
            mc_summary.update(prob_exceedance)

            mc_summary_df = pd.DataFrame.from_dict(mc_summary, orient='index', columns=['Value'])
            mc_summary_df.reset_index(inplace=True)
            mc_summary_df.rename(columns={'index': 'Metric'}, inplace=True)

            mc_summary_table = dbc.Table.from_dataframe(mc_summary_df.round(4), striped=True, bordered=True, hover=True)

            children = []
            if optimization_alert:
                children.append(optimization_alert)
            if threshold_alert:
                children.append(threshold_alert)

            children.extend([
                dcc.Graph(figure=path_fig),
                html.H3('Monte Carlo Simulation Summary'),
                html.Div(mc_summary_table, style={'fontSize': 14})
            ])

            return html.Div(children)

    elif active_tab == 'tab-stress-heatmap':
        # Create Heatmap for Stress Scenarios
        # Extract sectors and shocks
        sectors = sorted(set(sector for scenarios in stress_scenarios.values() for sector in scenarios['shocks'].keys()))

        # Create a matrix for heatmap
        heatmap_data = []
        scenario_names = []
        probabilities = []
        for scenario_id, details in stress_scenarios.items():
            scenario_names.append(details['name'])
            probabilities.append(details['probability'])
            row_data = [details['shocks'].get(sector, 0) for sector in sectors]
            heatmap_data.append(row_data)

        heatmap_df = pd.DataFrame(heatmap_data, columns=sectors, index=scenario_names)

        heatmap_fig = px.imshow(
            heatmap_df,
            labels=dict(x="Sector", y="Scenario", color="Shock Magnitude"),
            x=sectors,
            y=scenario_names,
            color_continuous_scale='RdBu',
            zmin=-0.5,
            zmax=0.5,
            aspect="auto"
        )

        heatmap_fig.update_layout(
            title='Stress Scenarios Heatmap',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=14),
            title_font_size=16,
            margin=dict(l=100, r=100, t=50, b=100)
        )
        heatmap_fig.update_xaxes(side="top")
        heatmap_fig.update_yaxes(autorange="reversed")

        children = []
        if optimization_alert:
            children.append(optimization_alert)

        children.extend([
            dcc.Graph(figure=heatmap_fig)
        ])

        return html.Div(children)

    else:
        return html.Div("No content available for this tab.")



if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8050))  # Use PORT if defined, else default to 8050
    app.run_server(host='0.0.0.0', port=port, debug=True)

