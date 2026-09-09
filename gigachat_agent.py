import os
import re
import json
from langchain_gigachat import GigaChat
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_classic.tools import StructuredTool
from pathlib import Path
from getpass import getpass

#---------------Settings---------------#
ENV_FILE = Path("./data/.env")
INI_FILE = Path("./ini.txt")

def load_ini():
    if INI_FILE.exists():
        with open(INI_FILE, "r") as ini:
            lines = ini.readlines()
        ini_params = {}
        for line in lines:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                ini_params[key] = value
        return ini_params.get("TIME"), ini_params.get("ITERATIONS")
    else:
        with open(INI_FILE, "w") as ini:
            ini.write("TIME=300\n")
            ini.write("ITERATIONS=15\n")
    return 300, 15

def load_credentials():
    if ENV_FILE.exists():
        with open(ENV_FILE, "r") as f:
            lines = f.readlines()
        creds = {}
        for line in lines:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                creds[key] = value
        return creds.get("GIGACHAT_CLIENT_ID"), creds.get("GIGACHAT_CREDENTIALS"), creds.get("GIGACHAT_SCOPE")
    return None, None, None

def save_credentials(client_id: str, credentials: str, scope: str = "GIGACHAT_API_PERS"):
    ENV_DIR.mkdir(exist_ok=True)
    with open(ENV_FILE, "w") as f:
        f.write(f"GIGACHAT_CLIENT_ID={client_id}\n")
        f.write(f"GIGACHAT_CREDENTIALS={credentials}\n")
        f.write(f"GIGACHAT_SCOPE={scope}\n")
    print("Credentials saved successfully!")

#---------------Setup Environment and Model---------------#
CLIENT_ID, SECRET_KEY, SCOPE = load_credentials()
PARAMETERS = load_ini()
if not CLIENT_ID:
    print("No credentials found. Please enter your GigaChat access token.")
    CLIENT_ID = input("Your client id: ")
    SECRET_KEY = getpass("Access Token: ")
    SCOPE = input("Scope (leave empty for GIGACHAT_API_PERS): ") or "GIGACHAT_API_PERS"
    save_credentials(CLIENT_ID, SECRET_KEY, SCOPE)
else:
    print("Credentials loaded from file.")

llm = GigaChat(
    credentials=SECRET_KEY,
    scope=SCOPE,
    verify_ssl_certs=False,
    model="GigaChat-2-Max",
    timeout=PARAMETERS[0]
)


#---------------Tools---------------#
def extract_json(text: str) -> dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return {"raw": text}

def parse_resume(text: str) -> dict:
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Ты — HR-ассистент. Проанализируй текст резюме и верни структурированные данные в формате JSON:
        {{
            "name": "Имя кандидата",
            "skills": ["список навыков"],
            "experience": "краткое описание опыта",
            "education": "образование",
            "achievements": ["список достижений"]
        }}
        Текст резюме:
        {text}
        """
    )
    chain = prompt | llm
    result = chain.invoke({"text": text})

    parsed = extract_json(result.content)
    return parsed

def match_with_vacancy(input_str: str) -> dict:
    try:
        data = json.loads(input_str)
        resume_data = data.get("resume")
        vacancy_desc = data.get("vacancy_desc")
        if not resume_data or not vacancy_desc:
            return {"raw": "Недостаточно данных: нужны resume и vacancy_desc"}
    except:
        return {"raw": "Ошибка парсинга входных данных. Ожидается JSON с resume и vacancy_desc."}
    
    prompt = PromptTemplate(input_variables=["resume", "vacancy"], template="""
        Ты — HR-эксперт. Сравни резюме с вакансией и верни JSON с оценкой:
        {{
            "match_percent": "число от 0 до 100",
            "strengths": ["сильная сторона 1", "сильная сторона 2"],
            "weaknesses": ["слабая сторона 1", "слабая сторона 2"],
            "recommendation": "подходит / не подходит / требуется дообучение"
        }}
        
        Резюме (в JSON): {resume}
        Вакансия: {vacancy}""")
    chain = prompt | llm
    result = chain.invoke({"resume": json.dumps(resume_data), "vacancy": vacancy_desc})
    try:
        return json.loads(result.content)
    except:
        return {result.content}


def generate_report(input_str: str) -> str:
    try:
        match_result = json.loads(input_str)
    except:
        match_result = extract_json(input_str)
        if "raw" in match_result:
            return "Не удалось извлечь данные из ответа."

    # Forming report
    if "match_percent" in match_result:
        percent = match_result.get('match_percent', 'N/A')
        strengths = match_result.get('strengths', [])
        weaknesses = match_result.get('weaknesses', [])
        recommendation = match_result.get('recommendation', 'N/A')
        
        report = f"""Результат анализа резюме
            Совпадение с вакансией: {percent}%
            Сильные стороны:
            {chr(10).join(['- ' + s for s in strengths]) if strengths else '—'}
            Слабые стороны:
            {chr(10).join(['- ' + s for s in weaknesses]) if weaknesses else '—'}
            Рекомендация: {recommendation}"""
        return report
    else:
        return "Не удалось сформировать отчёт: отсутствуют необходимые данные."

#---------------Agent Assemble---------------#

tools = [
    StructuredTool.from_function(
        name="ParseResume",
        func=parse_resume,
        description="Извлекает структурированную информацию из текста резюме. Вход: текст резюме."
    ),
    StructuredTool.from_function(
        name="MatchWithVacancy",
        func=match_with_vacancy,
        description="Сравнивает резюме с вакансией. Вход: JSON-строка с полями resume и vacancy_desc."
    ),
    StructuredTool.from_function(
        name="GenerateReport",
        func=generate_report,
        description="Формирует итоговый отчёт на основе результата сравнения. Вход: результат сравнения (словарь)."
    )
]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты — полезный ассистент. Используй доступные инструменты, чтобы проанализировать резюме и сравнить с вакансией."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,           # True for debug
    max_iterations=8,
    handle_parsing_errors=True,
    early_stopping_method="generate"
)

vacation_desc = input("Введите описание вакансии: ")
while vacation_desc == '':
    print('Описание вакансии не может быть пустым. Введите описание вакансии: ')
    vacation_desc = input("Введите описание вакансии: ")

resume = input("Приложите текст резюме: ")
while resume == '':
    print('Резюме не может быть пустым. Введите текст резюме: ')
    resume = input("Введите текст резюме: ")

query = f"Проанализируй это резюме и сравни с вакансией. Резюме: {resume} Вакансия: {vacation_desc}"
result = agent_executor.invoke({"input": query})
print('\n\nРекомендация GigaChat:\n' + result['output'])
