import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import plotly.express as px

def monte_carlo_option_pricing(S, K, T, r, sigma, simulations=100, option_type="call"):
    """
    Monte Carlo simulation for option pricing.

    Parameters:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate
        sigma: Volatility (standard deviation of returns)
        simulations: Number of Monte Carlo simulations
        option_type: "call" or "put"

    Returns:
        Estimated option price and simulated price paths.

    """
    dt = T / 252  # Assume 252 trading days in a year
    price_paths = np.zeros((simulations, 252))
    price_paths[:, 0] = S

    for t in range(1, 252):
        z = np.random.standard_normal(simulations)
        price_paths[:, t] = price_paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    if option_type == "call":
        payoffs = np.maximum(price_paths[:, -1] - K, 0)
    else:
        payoffs = np.maximum(K - price_paths[:, -1], 0)

    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price, price_paths

def plot_simulated_paths_interactive(price_paths):
    """
    Plot interactive simulated price paths for Monte Carlo simulation.

    Parameters:
        price_paths: Simulated price paths from Monte Carlo simulation.
    """
    # Combine the simulated price paths into a single DataFrame
    df_paths = pd.DataFrame(price_paths).T
    df_paths.columns = [f"Path {i+1}" for i in range(price_paths.shape[0])]

    # Create Plotly figure for simulated paths
    fig = px.line(
        df_paths,
        labels={"index": "Time Steps", "value": "Stock Price"},
        title="Simulated Price Paths",
    )
    fig.update_layout(
        legend_title_text="Simulation Paths",
        xaxis_title="Time Steps",
        yaxis_title="Stock Price",
        template="plotly_dark",
    )

    # Display the interactive plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
