import streamlit as st
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf

# --- Constants and Defaults ---
CURRENCY_PAIRS = {
    "USD/EUR (Eurozone)": "EURUSD=X",
    "USD/GBP (Britain)": "GBPUSD=X",
    "USD/JPY (Japan)": "JPYUSD=X",
    "USD/CAD (Canada)": "CADUSD=X",
    "USD/CHF (Switzerland)": "CHFUSD=X",
    "USD/AUD (Australia)": "AUDUSD=X",
    "USD/CNY (China)": "CNYUSD=X",
    "USD/RUB (Russia)": "RUBUSD=X",
}

DEFAULT_PAIR = "USD/EUR (Eurozone)"
DEFAULT_INPUTS = {
    'Local_Interest': 3.5,
    'US_Interest': 5.25,
    'Local_Inflation': 4.2,
    'US_Inflation': 3.1,
    'GDP_Growth': 2.4,
    'Trade_Balance': -3000,
    'Debt_to_GDP': 70.0
}

# --- Forex Prediction Model ---
def predict_fx(inputs: dict) -> float:
    """ Simple regression-like model for forex rate prediction. """
    rate = (
        1.0
        + 0.02 * (inputs['Local_Interest'] - inputs['US_Interest'])
        - 0.01 * (inputs['Local_Inflation'] - inputs['US_Inflation'])
        + 0.005 * inputs['GDP_Growth']
        + 0.00001 * inputs['Trade_Balance']
        - 0.001 * inputs['Debt_to_GDP']
    )
    # Prevent zero or negative FX rates
    return round(max(rate, 0.01), 4)

# --- Sentiment Analysis ---
@st.cache_resource
def get_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

def get_sentiment(text: str) -> float:
    """ Returns the compound sentiment score for the input text. """
    analyzer = get_sentiment_analyzer()
    try:
        score = analyzer.polarity_scores(text)['compound']
    except Exception:
        score = 0.0
    return score

# --- Currency Risk Calculator ---
def currency_risk(amount: float, volatility: float) -> float:
    """ Estimate maximum currency risk based on amount and expected volatility. """
    max_risk = amount * (volatility / 100)
    return round(max_risk, 2)

# --- Historical FX Data ---
@st.cache_data(show_spinner=False)
def get_historical_fx_data(symbol="EURUSD=X", period="1y"):
    df = yf.download(symbol, period=period, interval="1d", progress=False)
    # Always return a Series for charting
    if isinstance(df, pd.DataFrame) and not df.empty:
        if "Close" in df.columns:
            return df["Close"]
        else:
            # Fall back to the first column if 'Close' is missing
            return df.iloc[:, 0]
    return pd.Series(dtype="float64")

# --- Sensitivity Analysis ---
def sensitivity_analysis(inputs, param, values):
    results = []
    for v in values:
        test_inputs = inputs.copy()
        test_inputs[param] = v
        results.append(predict_fx(test_inputs))
    return results

# --- Streamlit App Layout ---
st.set_page_config(page_title="Forex Simulation App", layout="wide")
st.title("💱 Forex Simulation & Prediction App")

# --- Session State for Scenarios and Sentiments ---
if 'scenarios' not in st.session_state:
    st.session_state['scenarios'] = []
if 'sentiments' not in st.session_state:
    st.session_state['sentiments'] = []

# --- Tabs ---
tabs = st.tabs([
    "🔮 Prediction",
    "📈 History & Sensitivity",
    "📢 Alerts",
    "📰 Sentiment",
    "💼 Risk Calculator",
    "🗒️ Scenarios"
])

# --- Tab 1: Prediction ---
with tabs[0]:
    st.header("Forex Rate Prediction")
    st.markdown("Adjust the macroeconomic indicators to simulate and predict the exchange rate for your selected currency pair.")

    # Currency pair selector
    selected_pair = st.selectbox(
        "Currency Pair", list(CURRENCY_PAIRS.keys()), index=list(CURRENCY_PAIRS.keys()).index(DEFAULT_PAIR)
    )
    hist_symbol = CURRENCY_PAIRS[selected_pair]
    try:
        hist_fx = get_historical_fx_data(hist_symbol, "5d")
        current_fx_rate = float(hist_fx[-1]) if not hist_fx.empty else 0.0
    except Exception:
        current_fx_rate = 0.0

    # Infer country/region name from pair
    country = selected_pair.split("/")[1].split("(")[-1].replace(")", "").strip()
    inputs = {}
    col1, col2 = st.columns(2)
    with col1:
        inputs['Local_Interest'] = st.slider(f"{country} Interest Rate (%)", 0.0, 20.0, DEFAULT_INPUTS['Local_Interest'])
        inputs['Local_Inflation'] = st.slider(f"{country} Inflation Rate (%)", 0.0, 20.0, DEFAULT_INPUTS['Local_Inflation'])
        inputs['GDP_Growth'] = st.slider(f"{country} GDP Growth (%)", -10.0, 15.0, DEFAULT_INPUTS['GDP_Growth'])
    with col2:
        inputs['US_Interest'] = st.slider("US Interest Rate (%)", 0.0, 15.0, DEFAULT_INPUTS['US_Interest'])
        inputs['US_Inflation'] = st.slider("US Inflation Rate (%)", 0.0, 15.0, DEFAULT_INPUTS['US_Inflation'])
        inputs['Trade_Balance'] = st.slider(
            f"{country} Trade Balance (millions)", -10000, 10000, DEFAULT_INPUTS['Trade_Balance'], step=100
        )
        inputs['Debt_to_GDP'] = st.slider(f"{country} Public Debt to GDP (%)", 0.0, 150.0, DEFAULT_INPUTS['Debt_to_GDP'])

    predicted_rate = predict_fx(inputs)
    st.success(f"Predicted {selected_pair} rate: **{predicted_rate:,.4f}**")

    # Download prediction as CSV
    result_df = pd.DataFrame([inputs])
    result_df["Predicted Rate"] = predicted_rate
    result_df["Currency Pair"] = selected_pair
    st.download_button("Download Results as CSV", result_df.to_csv(index=False), file_name="fx_simulation.csv")

    # Save scenario
    note = st.text_input("Add a note to this scenario (optional):")
    if st.button("Save Scenario"):
        scenario = {"inputs": inputs.copy(), "rate": predicted_rate, "pair": selected_pair, "note": note}
        st.session_state['scenarios'].append(scenario)
        st.success("Scenario saved!")

# --- Tab 2: Historical Rates & Sensitivity ---
with tabs[1]:
    st.header("FX Rate History and Sensitivity Analysis")
    sub1, sub2 = st.columns(2)
    with sub1:
        st.subheader(f"{selected_pair} - 1 Year Historical Rates")
        hist_data = get_historical_fx_data(hist_symbol)
        if hist_data is not None and not hist_data.empty:
            # CORRECTED: Only call .rename("FX Rate") if hist_data is a Series
            if isinstance(hist_data, pd.Series):
                st.line_chart(hist_data.rename("FX Rate"))
            elif isinstance(hist_data, pd.DataFrame) and "Close" in hist_data.columns:
                st.line_chart(hist_data["Close"].rename("FX Rate"))
            else:
                st.line_chart(hist_data)
        else:
            st.warning("Historical data unavailable for this currency pair.")

    with sub2:
        st.subheader("Parameter Sensitivity Analysis")
        param = st.selectbox("Parameter", [
            f"{country} Interest Rate",
            "US Interest Rate",
            f"{country} Inflation Rate",
            "US Inflation Rate",
            f"{country} GDP Growth",
            f"{country} Trade Balance",
            f"{country} Public Debt to GDP"
        ], index=0)

        param_map = {
            f"{country} Interest Rate": 'Local_Interest',
            "US Interest Rate": 'US_Interest',
            f"{country} Inflation Rate": 'Local_Inflation',
            "US Inflation Rate": 'US_Inflation',
            f"{country} GDP Growth": 'GDP_Growth',
            f"{country} Trade Balance": 'Trade_Balance',
            f"{country} Public Debt to GDP": 'Debt_to_GDP'
        }
        param_key = param_map[param]

        if param_key in ["Local_Interest", "US_Interest", "Local_Inflation", "US_Inflation"]:
            values = np.linspace(0, 20, 40)
        elif param_key == "GDP_Growth":
            values = np.linspace(-10, 15, 40)
        elif param_key == "Trade_Balance":
            values = np.linspace(-10000, 10000, 40)
        else:  # Debt_to_GDP
            values = np.linspace(0, 150, 40)
        sens = sensitivity_analysis(inputs, param_key, values)
        sensitivity_df = pd.DataFrame({
            param: values,
            'Predicted Rate': sens
        }).set_index(param)
        st.line_chart(sensitivity_df)

# --- Tab 3: Alerts ---
with tabs[2]:
    st.header("Deviation Alert System")
    st.info(f"**Current FX Rate (approximate):** {current_fx_rate:,.4f}")
    st.info(f"**Model Prediction:** {predicted_rate:,.4f}")
    deviation = abs(predicted_rate - current_fx_rate) / predicted_rate if predicted_rate else 0
    if deviation > 0.02:
        st.error("⚠️ Alert: Exchange rate deviation exceeds 2% threshold!")
    else:
        st.success("✅ FX rate within acceptable deviation range.")

# --- Tab 4: Sentiment Analysis ---
with tabs[3]:
    st.header("Sentiment Analysis from Headlines")
    st.markdown("Enter a news headline or tweet about the economy to assess its sentiment impact on forex.")
    input_text = st.text_input(
        "Paste a news headline or tweet:", "ECB signals continued rate hikes as inflation remains high"
    )
    if st.button("Analyze Sentiment"):
        score = get_sentiment(input_text)
        st.session_state['sentiments'].append({'Headline': input_text, 'Score': score})
        if score >= 0.05:
            st.success(f"**Positive sentiment** (Score: {score:.2f})")
        elif score <= -0.05:
            st.error(f"**Negative sentiment** (Score: {score:.2f})")
        else:
            st.warning(f"**Neutral sentiment** (Score: {score:.2f})")

    if st.session_state['sentiments']:
        st.subheader("Sentiment Scores Over Time")
        df_sent = pd.DataFrame(st.session_state['sentiments'])
        st.line_chart(df_sent['Score'])
        st.dataframe(df_sent)

# --- Tab 5: Currency Risk Calculator ---
with tabs[4]:
    st.header("Currency Risk Calculator")
    st.markdown("Estimate potential loss based on amount and expected volatility of the exchange rate.")
    amount = st.number_input("Amount (in local currency)", min_value=0.0, value=10000.0, step=100.0)
    volatility = st.slider("Expected Volatility (%)", 0.0, 20.0, 3.0)
    risk = currency_risk(amount, volatility)
    st.info(f"**Estimated Currency Risk:** {risk:,.2f}")

# --- Tab 6: Scenarios ---
with tabs[5]:
    st.header("Saved Scenarios & Notes")
    if st.session_state['scenarios']:
        for i, scen in enumerate(st.session_state['scenarios']):
            with st.expander(f"Scenario {i+1} - {scen['pair']} ({scen['rate']:,.4f})"):
                st.write("**Inputs:**")
                st.json(scen['inputs'])
                st.write(f"**Predicted Rate:** {scen['rate']:,.4f}")
                st.write(f"**Currency Pair:** {scen['pair']}")
                if scen['note']:
                    st.write(f"**Note:** {scen['note']}")
    else:
        st.info("No scenarios saved yet.")
