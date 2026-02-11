@app.post("/telegram")
async def telegram_webhook(req: Request):
    data = await req.json()

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

