from stock_prediction import *
from blackScholes import *
from MonteCarloBasic import *
import streamlit as st
import plotly.express as px
from datetime import date, timedelta

def plot_with_streamlit(df):
    """
    Plot stock price with moving averages and RSI for Streamlit.
    """
    # Calculate moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = calculate_rsi(df['Close'])

    # Plot stock price with moving averages
    fig_price = px.line(
        df,
        x=df.index,
        y=['Close', 'MA20', 'MA50', 'MA200'],
        labels={"value": "Price", "index": "Date"},
        title="Stock Price with Moving Averages",
    )
    fig_price.update_layout(
        legend_title_text="Legend",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # Plot RSI
    fig_rsi = px.line(
        df,
        x=df.index,
        y='RSI',
        labels={"value": "RSI", "index": "Date"},
        title="Relative Strength Index (RSI)",
    )
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_rsi.update_layout(
        xaxis_title="Date",
        yaxis_title="RSI",
        template="plotly_dark",
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

def main():
    st.title("Stock Analysis and Option Pricing")

    # User choice
    model_choice = st.sidebar.radio("Choose Pricing Model", ["Black-Scholes", "Monte Carlo"])

    # User inputs
    ticker = st.sidebar.text_input("Stock Ticker Symbol", "AAPL")
    strike_price = st.sidebar.number_input("Strike Price (K)", value=250.0)
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05)
    time_to_maturity = st.sidebar.number_input("Time to Expiry (T, in years)", value=1.0)
    volatility = st.sidebar.number_input("Volatility (σ)", value=0.2)
    option_type = st.sidebar.selectbox("Option Type", ["call", "put"])

    if model_choice == "Monte Carlo":
        simulations = st.sidebar.number_input("Number of Simulations", value=100, min_value=50, step=100)

    # Fetch stock data
    st.write("Fetching stock data...")
    end_date = date.today()
    start_date = end_date - timedelta(days=5 * 365)
    df = get_stock_data(ticker, start=start_date, end=end_date)

    if df is not None:
        df = df.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')
        st.write(f"### Historical Data for {ticker}", df.tail())

        # Plot stock price with moving averages and RSI
        st.write("### Stock Price with Moving Averages")
        plot_with_streamlit(df)

        # Prepare features
        X, y = prepare_data(df)

        # Train model
        model, train_score, test_score = train_model(X, y)

        # Predict next price
        predicted_price = model.predict(X.iloc[[-1]].values)[0]
        st.write(f"Predicted Next Close Price: ${predicted_price:.2f}")

        if model_choice == "Black-Scholes":
            # Calculate Black-Scholes option price and Greeks
            option_price = calculate_black_scholes(
                S=predicted_price,
                K=strike_price,
                r=risk_free_rate,
                T=time_to_maturity,
                sigma=volatility,
                option_type=option_type,
            )
            delta, gamma, theta, vega, rho = calculate_greeks(
                S=predicted_price,
                K=strike_price,
                r=risk_free_rate,
                T=time_to_maturity,
                sigma=volatility,
                option_type=option_type,
            )
            st.write(f"Option Price: ${option_price:.2f}")
            st.write(f"Greeks:\nDelta: {delta:.2f}, Gamma: {gamma:.2f}, Theta: {theta:.2f}, Vega: {vega:.2f}, Rho: {rho:.2f}")

        elif model_choice == "Monte Carlo":
            # Calculate Monte Carlo option price
            monte_carlo_price, price_paths = monte_carlo_option_pricing(
                S=predicted_price,
                K=strike_price,
                T=time_to_maturity,
                r=risk_free_rate,
                sigma=volatility,
                simulations=simulations,
                option_type=option_type,
            )
            st.write(f"Monte Carlo Option Price: ${monte_carlo_price:.2f}")
            st.write("### Simulated Price Paths")
            plot_simulated_paths_interactive(price_paths)

if __name__ == "__main__":
    main()
