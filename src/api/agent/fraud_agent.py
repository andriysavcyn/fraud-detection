from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from src.api.agent.tools import get_transaction_risk, get_user_profile, get_fraud_patterns
import os

load_dotenv()

class FraudAnalysisReport(BaseModel):
    verdict: str
    confidence: str
    summary: str
    recommend_action: str

SYSTEM_PROMPT = """Ти — старший аналітик з фінансових злочинів у банку.
Твоя задача: аналізувати транзакції на предмет шахрайства.

Коли отримуєш транзакцію:
1. Завжди спочатку викликай get_transaction_risk щоб отримати ML score
2. Потім get_user_profile щоб розуміти контекст юзера
3. Потім get_fraud_patterns щоб ідентифікувати схему шахрайства
4. На основі ВСІХ даних — формулюй чіткий вердикт

Твої відповіді мають бути:
- Конкретними (називай цифри, факти)
- Зрозумілими для нетехнічного аналітика
- З чіткою рекомендацією дії

Відповідай українською мовою.

ВАЖЛИВО: Якщо get_transaction_risk повертає помилку — НЕ викликай його повторно.
Продовжуй аналіз на основі get_user_profile та get_fraud_patterns.
Сформулюй вердикт з наявних даних."""

def create_fraud_agent():
    llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
)

    tools = [get_transaction_risk, get_user_profile, get_fraud_patterns]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
    )

class FraudAnalystChat:
    def __init__(self):
        self.agent = create_fraud_agent()
        self.chat_history = []

    def analyze(self, user_message: str) -> str:
        response = self.agent.invoke({
            "input": user_message,
            "chat_history": self.chat_history,
        })

        # зберігаємо історію для follow-up питань
        self.chat_history.extend([
            HumanMessage(content=user_message),
            AIMessage(content=response["output"]),
        ])

        return response["output"]