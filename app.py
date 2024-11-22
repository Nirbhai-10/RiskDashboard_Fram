# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import statsmodels.api as sm

# Initialize the Dash app with a light Bootstrap theme
external_stylesheets = [dbc.themes.FLATLY]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Advanced Portfolio Risk Dashboard"

# Increase the number of stocks to 20 for better diversification
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
    return result.x

# Define function to calculate Black-Litterman weights
def calculate_black_litterman_weights(returns):
    # Assume market equilibrium returns are the mean returns
    cov_matrix = returns.cov()
    market_weights = np.ones(len(returns.columns)) / len(returns.columns)
    delta = 2.5
    tau = 0.05
    pi = delta * np.dot(cov_matrix, market_weights)

    # Investor views
    Q = np.array([0.02])  # Expected excess return of 2% on the first asset
    P = np.zeros((1, len(returns.columns)))
    P[0, 0] = 1  # The view is on the first asset
    omega = np.array([[0.0001]])

    # Black-Litterman formula
    inv_cov = np.linalg.inv(tau * cov_matrix)
    inv_omega = np.linalg.inv(omega)
    mu_bl = np.linalg.inv(inv_cov + np.dot(P.T, np.dot(inv_omega, P)))
    mu_bl = np.dot(mu_bl, (np.dot(inv_cov, pi) + np.dot(P.T, np.dot(inv_omega, Q))))
    weights = np.dot(np.linalg.inv(cov_matrix), mu_bl) / delta
    weights = np.clip(weights, 0, None)
    weights /= np.sum(weights)
    return weights

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
    return result.x

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
                    {'label': 'Maximum Sharpe Ratio', 'value': 'MaxSharpe'}
                ],
                value='PCA'
            ),
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
            html.Label("Enter Portfolio Value:"),
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
            dbc.Button('Update', id='update-button', n_clicks=0, color='primary')
        ], width=3, style={'padding': '20px'}),
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label='Risk Dashboard', tab_id='tab-risk'),
                dbc.Tab(label='Portfolio Weights', tab_id='tab-weights'),
                dbc.Tab(label='Return Distribution', tab_id='tab-returns'),
                dbc.Tab(label='Risk Contributions', tab_id='tab-contributions'),
                dbc.Tab(label='Risk Metrics Table', tab_id='tab-metrics')
            ], id='tabs', active_tab='tab-risk'),
            html.Div(id='tab-content')
        ], width=9)
    ])
], fluid=True)

@app.callback(
    Output('tab-content', 'children'),
    [Input('update-button', 'n_clicks'),
     Input('tabs', 'active_tab'),
     Input('confidence-level', 'value')],
    [State('time-horizon', 'value'),
     State('weight-method', 'value'),
     State('var-method', 'value'),
     State('portfolio-value', 'value')]
)
def render_content(n_clicks, active_tab, confidence_level, time_horizon, weight_method, var_method, portfolio_value):
    # Update weights based on the selected method
    if weight_method == 'PCA':
        weights = calculate_pca_weights(returns)
    elif weight_method == 'RiskParity':
        weights = calculate_risk_parity_weights(returns)
    elif weight_method == 'BlackLitterman':
        weights = calculate_black_litterman_weights(returns)
    elif weight_method == 'MaxSharpe':
        weights = calculate_max_sharpe_weights(returns)
    else:
        weights = np.ones(len(returns.columns)) / len(returns.columns)

    # Convert confidence level to decimal
    confidence_level = confidence_level / 100.0

    # Calculate VaR and ES at the selected confidence level
    var_percent, es_percent, portfolio_returns = calculate_var_es_single(
        returns, weights, confidence_level, time_horizon, method=var_method
    )

    # Calculate VaR and ES in monetary terms
    var_monetary = var_percent * portfolio_value
    es_monetary = es_percent * portfolio_value

    # Calculate additional risk metrics
    risk_metrics = calculate_risk_metrics(returns, weights, portfolio_returns, time_horizon)

    if active_tab == 'tab-risk':
        # Create cards to display VaR and ES
        var_card = dbc.Card(
            [
                dbc.CardHeader(f"Value at Risk (VaR) at {int(confidence_level * 100)}% Confidence Level"),
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
                dbc.CardHeader(f"Expected Shortfall (ES) at {int(confidence_level * 100)}% Confidence Level"),
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
            xaxis_title='Portfolio Returns',
            yaxis_title='Cumulative Probability',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=14),
            legend=dict(font_size=12),
            title_font_size=16
        )

        return html.Div([
            cards,
            dcc.Graph(figure=risk_fig)
        ])

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
            title_font_size=16
        )
        weights_fig.update_xaxes(tickfont_size=12)
        weights_fig.update_yaxes(title_font_size=14)
        return dcc.Graph(figure=weights_fig)

    elif active_tab == 'tab-returns':
        # Create the return distribution histogram
        hist_data = portfolio_returns

        # Create the histogram figure
        hist_fig = px.histogram(
            x=hist_data,
            nbins=30,
            title='Portfolio Return Distribution',
            labels={'x': 'Returns', 'y': 'Frequency'},
            opacity=0.75,
            color_discrete_sequence=['#636EFA']
        )

        hist_fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=14),
            bargap=0.2,
            legend=dict(font_size=12),
            title_font_size=16
        )
        hist_fig.update_xaxes(title_font_size=14)
        hist_fig.update_yaxes(title_font_size=14)

        # Display VaR and ES values
        var_text = f"Value at Risk (VaR) at {int(confidence_level * 100)}% confidence level: {var_percent:.2%}"
        es_text = f"Expected Shortfall (ES) at {int(confidence_level * 100)}% confidence level: {es_percent:.2%}"

        return html.Div([
            html.Div([
                html.P(var_text, style={'fontSize': 16}),
                html.P(es_text, style={'fontSize': 16})
            ], style={'textAlign': 'center', 'marginBottom': 20}),
            dcc.Graph(figure=hist_fig)
        ])

    elif active_tab == 'tab-contributions':
        # Risk Contributions Treemap
        cov_matrix = returns.cov()
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_contrib = cov_matrix.dot(weights) / portfolio_std
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
            title_font_size=16
        )

        return dcc.Graph(figure=treemap_fig)

    elif active_tab == 'tab-metrics':
        # Create a table summarizing risk metrics
        metrics_df = pd.DataFrame.from_dict(risk_metrics, orient='index', columns=['Value'])
        metrics_df.reset_index(inplace=True)
        metrics_df.rename(columns={'index': 'Metric'}, inplace=True)
        metrics_table = dbc.Table.from_dataframe(metrics_df.round(4), striped=True, bordered=True, hover=True)
        return html.Div([
            html.H3('Risk Metrics Summary'),
            html.Div(metrics_table, style={'fontSize': 14})
        ])

    else:
        return html.Div("No content available for this tab.")

if __name__ == '__main__':
    app.run_server(debug=True)
