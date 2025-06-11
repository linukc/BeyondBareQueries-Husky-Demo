import os
import yaml
import json
import logging
from PIL import Image, ImageDraw, ImageFont
import requests
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
# import TinyTag

import subprocess

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

with open("../secrets.yaml") as stream:
    secrets = yaml.safe_load(stream)

# Конфигурация
TOKEN = secrets["TOKEN"]
SALUTE_SPEECH_API_KEY = secrets["SALUTE_SPEECH_API_KEY"]
SALUTE_SPEECH_URL = "https://smartspeech.sber.ru/rest/v1/speech:recognize"

YANDEX_FOLDER_ID=secrets["YANDEX_FOLDER_ID"]
YANDEX_API_KEY=secrets["YANDEX_API_KEY"]


def get_salute_speech_token() -> str:

    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    payload={
    'scope': 'SALUTE_SPEECH_PERS'
    }
    headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Accept': 'application/json',
    'RqUID': '603bdf01-b8b8-4f77-a1a5-a99b0abe7e74',
    'Authorization': f'Basic {SALUTE_SPEECH_API_KEY}'
    }

    response = requests.request("POST", url, headers=headers, data=payload, verify=False)

    if response.status_code == 200:
        return response.json().get('access_token', 'Empty access token field')
    else:
        err = f"Ошибка Salute Speech Access Token: {response.status_code} - {response.text}"
        logger.error(err)
        raise Exception(f"Ошибка при попытке обновить Salute Speech Access Token: {err}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    welcome_text = (
        "Привет! Я бот для отправки запроса к пайплайну BBQ.\n\n"
        "Отправьте мне текстовый запрос на русском языке\n"
        "и я переведу и передам его роботу для обработки.\n"
    )
    await update.message.reply_text(welcome_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    user_text = update.message.text
    context.user_data['text_query'] = user_text
    await update.message.reply_text("Текст запроса сохранён.")

    # Проверяем, есть ли текстовый запрос
    if 'text_query' in context.user_data:
        await process_request(update, context)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик голосовых сообщений"""
    try:
        # Получаем голосовое сообщение
        voice_file = await update.message.voice.get_file()
        print(voice_file)
        
        # Скачиваем файл
        voice_path = f"voice_{update.update_id}.ogg"
        await voice_file.download_to_drive(voice_path)
        
        # Преобразуем в текст
        await update.message.reply_text("Обрабатываю ваше сообщение...")
        text = await speech_to_text(voice_path)

        # Отправляем результат
        await update.message.reply_text(f"Текст запроса сохранён. Распознанный текст:\n\n{text}")

        context.user_data['text_query'] = text

        # Удаляем временный файл
        os.remove(voice_path)
        # Проверяем, есть ли текстовый запрос
        if 'text_query' in context.user_data:
            await process_request(update, context)

    except Exception as e:
        logger.error(f"Ошибка при обработке голоса: {e}")
        await update.message.reply_text(f"Произошла ошибка при обработке голосового сообщения: {e}")

async def speech_to_text(audio_path: str) -> str:
    
    """Преобразование голоса в текст с помощью Salute Speech"""
    SALUTE_ACCESS_TOKEN = get_salute_speech_token()
    headers = {
        "Authorization": f"Bearer {SALUTE_ACCESS_TOKEN}",
        "Content-Type": "audio/ogg;codecs=opus"
    }
    
    with open(audio_path, 'rb') as audio_file:
        response = requests.post(
            SALUTE_SPEECH_URL,
            headers=headers,
            data=audio_file,
            verify=False
        )
    
    if response.status_code == 200:
        return response.json().get('result', 'Текст не распознан')
    else:
        err = f"Ошибка Salute Speech API: {response.status_code} - {response.text}"
        logger.error(err)
        raise Exception(f"Ошибка при распознавании речи: {err}")

async def process_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка запроса (изображение + текст)"""
    try:
        text_query = context.user_data['text_query']

        if isinstance(text_query, list):
            if text_query:  # Проверка, что список не пустой
                text_query = text_query[0]
            else:
                text_query = None
        
        text_query = await translate(text_query)
        # Отправляем уведомление о начале обработки
        await update.message.reply_text(f"Обрабатываю ваш запрос: {text_query}")
        
        url = 'https://0.0.0.0:4444/first'
        response = requests.post(url, data=text_query, verify=False).json()
        print(response)
        await update.message.reply_text(f"Робот выбрал релевантные объекты: {response['message']}")
        
        url = 'https://0.0.0.0:4444/second'
        response = requests.post(url, data=text_query, verify=False).json()
        text = str(response['message']).replace('\\n', '').replace('\\', '')
        if '{' in text and '}' in text:
            json_part = text[text.find('{'):text.find('}')+1]
            print("json_part: ", json_part)
            try:
                data = json.loads(json_part)
                print("data: ", data)
                explanation = data['explanation']
                id = data['id']
            except Exception as e:
                print("error: ", e)
                pass

        select = f"I select the object with id {id}"
        select = select + explanation

        select = await translate(select, target_language='ru')

        await update.message.reply_text(f"Финальный ответ робота: {select}")



        # Отправляем результаты пользователю
        await update.message.reply_text(f"Робот готов получать новые запросы.")
            

        context.user_data.pop('text_query', None)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        await update.message.reply_text(f"Произошла ошибка при обработке вашего запроса: {e}.")


async def translate(text,
              target_language='en'):

    print("Перевожу", text)
    body = {
        "targetLanguageCode": target_language,
        "texts": text,
        "folderId": YANDEX_FOLDER_ID,
    }

    headers = {
        "Content-Type": "application/json",
        'Authorization': f'Api-Key {YANDEX_API_KEY}',
    }

    response = requests.post('https://translate.api.cloud.yandex.net/translate/v2/translate',
        json=body,
        headers=headers
    )

    print(response.json())
    if "translations" in response.json(): 
        return response.json()["translations"][0]["text"]
    else:
        return None

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ошибок"""
    logger.error(f"Ошибка при обработке обновления {update}: {context.error}")
    if update.message:
        await update.message.reply_text(f"Произошла ошибка. Пожалуйста, попробуйте еще раз.{update}: {context.error}")

def main():
    """Запуск бота"""
    application = Application.builder().token(TOKEN).build()
    
    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_error_handler(error_handler)
    
    # Запуск бота
    application.run_polling()

if __name__ == '__main__':

    main()


# reqs
# pip install python-telegram-bot pillow  transformers requests

# start
# python bot.py
