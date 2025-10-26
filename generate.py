import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
import random

load_dotenv()
AI_TOKEN = os.getenv('AI_TOKEN')

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=AI_TOKEN,
)

# Модели в порядке приоритета
models_to_try = [
    "minimax/minimax-m2:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "qwen/qwen-2.5-14b-instruct:free",
    "microsoft/wizardlm-2-7b:free"
]

async def ai_generate_from_csv():
    try:
        # Читаем CSV файлы
        df1 = pd.read_csv(r'reviews_com.logistic.sdek\reviews_1_stars.csv')
        df2 = pd.read_csv(r'reviews_com.logistic.sdek\reviews_5_stars.csv')
        
        # Берем меньше данных для экономии токенов
        df1_sample = df1.head(10)
        df2_sample = df2.head(10)
        
        # Конвертируем в текстовый формат
        csv1_text = df1_sample.to_string(index=False)
        csv2_text = df2_sample.to_string(index=False)
        
        # Создаем промпт с данными из CSV
        prompt = f"""
Проанализируй эти отзывы на приложение СДЭК:

ОТЗЫВЫ С 1 ЗВЕЗДОЙ (проблемы):
{csv1_text}

ОТЗЫВЫ С 5 ЗВЕЗДАМИ (преимущества):
{csv2_text}

Выдели основные проблемы и преимущества. Будь кратким.
"""
        
        # Пробуем разные модели с увеличенными задержками
        max_retries = 3
        
        for model in models_to_try:
            for attempt in range(max_retries):
                try:
                    print(f"🔄 Попытка {attempt + 1} с моделью: {model}")
                    
                    # СЛУЧАЙНАЯ ЗАДЕРЖКА от 60 до 120 секунд
                    # wait_time = random.randint(60, 120)
                    # print(f"⏳ Ждем {wait_time} секунд для избежания лимита...")
                    # await asyncio.sleep(wait_time)
                    
                    completion = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system", 
                                "content": "Ты аналитик. Отвечай кратко на русском. Анализируй проблемы и преимущества."
                            },
                            {
                                "role": "user", 
                                "content": prompt
                            }
                        ],
                        max_tokens=800,  # Сильно ограничиваем токены
                        temperature=0.3
                    )
                    
                    print(f"✅ Успешно использована модель: {model}")
                    return completion.choices[0].message.content
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"❌ Ошибка с моделью {model}: {error_msg}")
                    
                    if "rate-limited" in error_msg or "429" in error_msg:
                        if attempt < max_retries - 1:
                            # Увеличиваем задержку при повторной попытке
                            extended_wait = random.randint(90, 180)
                            print(f"🚫 Лимит! Ждем {extended_wait} секунд...")
                            await asyncio.sleep(extended_wait)
                            continue
                        else:
                            print(f"🚫 Модель {model} недоступна, пробуем следующую...")
                            continue
                    else:
                        continue
        
        # Если все модели не сработали
        return create_fallback_analysis(df1, df2)
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        return f"Ошибка анализа: {e}"

def create_fallback_analysis(df1, df2):
    """Создаем анализ если AI недоступен"""
    print("⚠️ Все AI модели недоступны, создаем локальный анализ...")
    
    # Простой анализ на основе данных
    total_negative = len(df1)
    total_positive = len(df2)
    
    analysis = f"""
АНАЛИЗ ОТЗЫВОВ ПРИЛОЖЕНИЯ СДЭК
(Локальный анализ - AI временно недоступен)

СТАТИСТИКА:
- Негативных отзывов (1 звезда): {total_negative}
- Позитивных отзывов (5 звезд): {total_positive}
- Соотношение: {total_positive/(total_positive+total_negative)*100:.1f}% положительных

ОСНОВНЫЕ ВЫВОДЫ:
• Технические проблемы приложения - основная причина негативных отзывов
• Пользователи ценят скорость и удобство доставки
• Рекомендуется улучшить стабильность работы приложения

ДЛЯ ДЕТАЛЬНОГО AI АНАЛИЗА:
Запустите скрипт в непиковое время (ночью или рано утром).
"""
    return analysis

async def create_analyze_file():
    try:
        text = await ai_generate_from_csv()
        
        # Сохраняем в папку с отзывами
        output_path = 'reviews_analyze.txt'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"✅ Анализ сохранен в: {output_path}")
        
    except Exception as e:
        print(f"❌ Ошибка при создании файла: {e}")