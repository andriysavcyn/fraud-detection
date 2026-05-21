from src.api.agent.fraud_agent import FraudAnalystChat

chat = FraudAnalystChat()

result = chat.analyze("""
Проаналізуй цю транзакцію для юзера user_001:
{
    "amount": 2400.0,
    "hour": 3,
    "is_new_location": 1,
    "merchant_category": "online",
    "user_avg_amount": 45.0,
    "transactions_last_hour": 5,
    "is_foreign_country": 1,
    "card_present": 0
}
""")

print("\n" + "="*50)
print("ВІДПОВІДЬ АГЕНТА:")
print(result)
print("="*50)

# Follow-up питання — тест multi-turn
followup = chat.analyze("Які конкретні дії рекомендуєш? Чи варто одразу блокувати?")
print("\nFOLLOW-UP:")
print(followup)