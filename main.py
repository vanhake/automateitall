import os
from fastapi import FastAPI, Request
from telegram import Bot
from openai import OpenAI
import time
import json

RATE_LIMIT = 10        # max. Anfragen
RATE_WINDOW = 60       # pro 60 Sekunden
user_requests = {}    # user_id → [timestamps]

# ENV
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("OPENAI_API_KEY not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

# Telegram User IDs, die den Bot nutzen dürfen
ALLOWED_USERS = {
    #123456789,   # du
    #987654321    # weiterer Nutzer
}

bot = Bot(token=TELEGRAM_TOKEN)
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

@app.post("/telegram")
async def telegram_webhook(req: Request):
    data = await req.json()
    print(json.dumps(data, indent=2))

    if "message" not in data:
        return {"ok": True}

    user_id = data["message"]["from"]["id"]
    chat_id = data["message"]["chat"]["id"]
    text = data["message"].get("text", "")

    # 1️⃣ Whitelist
    if user_id not in ALLOWED_USERS:
        await bot.send_message(chat_id, "⛔ Zugriff nicht erlaubt.")
        return {"ok": True}

    # 2️⃣ Rate Limit
    now = time.time()
    timestamps = user_requests.get(user_id, [])
    timestamps = [t for t in timestamps if now - t < RATE_WINDOW]

    if len(timestamps) >= RATE_LIMIT:
        await bot.send_message(chat_id, "⏳ Rate Limit erreicht.")
        return {"ok": True}

    timestamps.append(now)
    user_requests[user_id] = timestamps

    # 3️⃣ Input Limit
    if len(text) > 2000:
        await bot.send_message(chat_id, "✂️ Nachricht zu lang.")
        return {"ok": True}

    # 4️⃣ LLM Call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Du bist ein hilfreicher KI-Assistent."},
            {"role": "user", "content": text}
        ],
        max_tokens=500
    )

    await bot.send_message(chat_id, response.choices[0].message.content)
    return {"ok": True}
