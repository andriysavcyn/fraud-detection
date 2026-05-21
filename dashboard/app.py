import json
import random
import requests
import streamlit as st

FASTAPI_URL = "https://127.0.0.1:8000"

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

import sys
sys.path.append(".")
from src.api.agent.fraud_agent import FraudAnalystChat

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "agent" not in st.session_state:
    st.session_state.agent = FraudAnalystChat()

if "last_transaction" not in st.session_state:
    st.session_state.last_transaction = None

def generate_random_transaction(fraud: bool = False):
    if fraud:
        return{
            "amount": round(random.uniform(500, 5000), 2),
            "hour": random.choice([0, 1, 2, 3, 4, 22, 23]),
            "is_new_location": 1,
            "merchant_category": random.choice(["online", "travel"]),
            "user_avg_amount": round(random.uniform(20, 80), 2),
            "transactions_last_hour": random.randint(4, 9),
            "is_foreign_country": 1,
            "card_present": 0,
        }
    else:
        return {
            "amount": round(random.uniform(10, 200), 2),
            "hour": random.randint(9, 21),
            "is_new_location": 0,
            "merchant_category": random.choice(["grocery", "restaurant", "gas"]),
            "user_avg_amount": round(random.uniform(30, 150), 2),
            "transactions_last_hour": random.randint(0, 2),
            "is_foreign_country": 0,
            "card_present": 1,
        }
    
def call_predict(transaction: dict):
    try:
        r = requests.post("http://127.0.0.1:8000/predict", json=transaction, timeout=5)
        return r.json()
    except Exception as e:
        st.error(f"FastAPI недоступний: {e}")
        return None
    
st.title("Fraud Detection Dashboard")
st.caption("Hybrid ML + LLM Agent System")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Live Transaction Simulation")

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        if st.button("Legity", use_container_width=True):
            tx = generate_random_transaction(fraud=False)
            st.session_state.last_transaction = tx

    with sim_col2:
        if st.button("Suspected", use_container_width=True):
            tx = generate_random_transaction(fraud=True)
            st.session_state.last_transaction = tx

    with sim_col1:
        if st.button("Accident", use_container_width=True):
            tx = generate_random_transaction(fraud=random.choice([True, False]))
            st.session_state.last_transaction = tx

    with st.expander("✏️ Ввести транзакцію вручну"):
        m_amount = st.number_input("Amount ($)", value=100.0, min_value=0.1)
        m_hour = st.slider("Hour", 0, 23, 12)
        m_category = st.selectbox("Category", ["grocery", "restaurant", "gas", "online", "travel"])
        m_avg = st.number_input("User avg amount ($)", value=50.0)
        m_txn_hour = st.number_input("Transactions last hour", value=1, min_value=0)
        m_new_loc = st.checkbox("New location")
        m_foreign = st.checkbox("Foreign country")
        m_card = st.checkbox("Card present", value=True)

        if st.button("📤 Надіслати", use_container_width=True):
            st.session_state.last_transaction = {
                "amount": m_amount,
                "hour": m_hour,
                "is_new_location": int(m_new_loc),
                "merchant_category": m_category,
                "user_avg_amount": m_avg,
                "transactions_last_hour": int(m_txn_hour),
                "is_foreign_country": int(m_foreign),
                "card_present": int(m_card),
            }
            st.rerun()

    if st.session_state.last_transaction:
        tx = st.session_state.last_transaction

        with st.spinner("Надсилання транзакції в ML модель..."):
            result = call_predict(tx)

        if result:
            st.divider()
            st.subheader("📊 ML Model Result")

            risk = result["risk_level"]
            prob = result["fraud_probability"]

            # колір залежно від ризику
            color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk, "⚪")

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Fraud Probability", f"{prob:.1%}")
            metric_col2.metric("Risk Level", f"{color} {risk}")
            metric_col3.metric("Verdict", "🚫 BLOCK" if result["is_fraud"] else "✅ APPROVE")

            st.progress(prob)

            st.subheader("🔬 Top Risk Factors (SHAP)")
            for factor in result["top_risk_factors"]:
                impact = factor["shap_impact"]
                bar_color = "🔴" if impact > 0 else "🟢"
                st.write(
                    f"{bar_color} **{factor['feature']}** — "
                    f"value: `{factor['value']:.2f}` | "
                    f"impact: `{impact:+.3f}`"
                )

            with st.expander("📋 Деталі транзакції"):
                st.json(tx)

            if st.button("🤖 Запитати агента про цю транзакцію", use_container_width=True):
                question = f"""Проаналізуй цю транзакцію для юзера user_001:
{json.dumps(tx, ensure_ascii=False)}

ML модель дала fraud probability: {prob:.1%}, risk level: {risk}.
Поясни чому і що робити."""
                
                st.session_state.chat_history.append(
                    {"role": "user", "content": question}
                )
                with st.spinner("Агент аналізує..."):
                    response = st.session_state.agent.analyze(question)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )
                st.rerun()

with col_right:
    st.subheader("🤖 Fraud Analyst Agent")
    st.caption("Запитуй агента про будь-яку транзакцію")

    # історія чату
    chat_container = st.container(height=500)
    with chat_container:
        if not st.session_state.chat_history:
            st.info("👋 Симулюй транзакцію зліва або напиши питання нижче")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])


    user_input = st.chat_input("Запитай агента...")
    if user_input:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )
        with st.spinner("Агент думає..."):
            response = st.session_state.agent.analyze(user_input)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )
        st.rerun()

    if st.button("🗑️ Очистити чат", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.agent = FraudAnalystChat()
        st.rerun()           
            