import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()
AI_TOKEN = os.getenv('AI_TOKEN')

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=AI_TOKEN,
)

async def ai_generate_from_csv():
    # Читаем CSV файлы
    df1 = pd.read_csv(r'reviews_com.logistic.sdek\reviews_1_stars.csv')
    df2 = pd.read_csv(r'reviews_com.logistic.sdek\reviews_5_stars.csv')
    
    # Конвертируем в текстовый формат
    csv1_text = df1.to_string(index=False)
    csv2_text = df2.to_string(index=False)
    
    # Создаем промпт с данными из CSV
    prompt = f"""
Данные из первого файла с 1 звездой:
{csv1_text}

Данные из второго файла с 5 звездами:
{csv2_text}
"""
    
    models_to_try = [
        "deepseek/deepseek-r1:free",
        "microsoft/wizardlm-2-8x22b:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "qwen/qwen-2.5-32b-instruct:free"        
    ]
    
    # Пробуем разные модели с повторными попытками
    max_retries = 2
    for model in models_to_try:
        for attempt in range(max_retries):
            try:
                print(f"🔄 Попытка {attempt + 1} с моделью: {model}")
                
                # Добавляем задержку между запросами
                if attempt > 0:
                    wait_time = 30  # 30 секунд между повторами
                    print(f"⏳ Ждем {wait_time} секунд...")
                    await asyncio.sleep(wait_time)
                
                completion = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "developer", "content": "Ты опытный аналитик мобильных приложений. Анализируй отзывы кратко и по делу .Выдели главные проблемы (файл с 1 звездой) и приемущества(файл с 5 звездами)."},
                        {"role": "user", "content": prompt}
                    ],
                    timeout= 60,
                    max_tokens=1500,  # Ограничиваем длину ответа
                    temperature=0.7
                )
                
                return completion.choices[0].message.content
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Ошибка с моделью {model}: {error_msg}")
                
                if "rate-limited" in error_msg or "429" in error_msg:
                    if attempt < max_retries - 1:
                        continue  # Пробуем еще раз с той же моделью
                    else:
                        continue  # Переходим к следующей модели
                else:
                    continue  # Переходим к следующей модели

async def create_analyze_file():
    text = await ai_generate_from_csv()
    with open(f'reviews_analyze.txt', 'w', encoding='utf-8') as f:
            f.write(text)