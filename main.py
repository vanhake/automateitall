import os
import logging
from fastapi import FastAPI, Request, HTTPException
from telegram import Bot
from telegram.error import TelegramError
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
import time
import json
from typing import Dict, List, Set
from datetime import datetime

# ============================================================================
# LOGGING KONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# RATE LIMITING KONFIGURATION
# ============================================================================
RATE_LIMIT = 10        # max. Anfragen
RATE_WINDOW = 60       # pro 60 Sekunden
MAX_INPUT_LENGTH = 2000
MAX_TOKENS = 500

user_requests: Dict[int, List[float]] = {}

# ============================================================================
# UMGEBUNGSVARIABLEN LADEN UND VALIDIEREN
# ============================================================================
def load_allowed_users() -> Set[int]:
    """Lädt erlaubte User IDs aus Umgebungsvariable."""
    raw = os.getenv("ALLOWED_USERS", "")
    if not raw:
        logger.warning("⚠️ ALLOWED_USERS ist leer - kein User hat Zugriff!")
        return set()
    
    try:
        users = {int(uid.strip()) for uid in raw.split(",") if uid.strip()}
        logger.info(f"✅ {len(users)} erlaubte User geladen: {users}")
        return users
    except ValueError as e:
        logger.error(f"❌ Fehler beim Parsen von ALLOWED_USERS: {e}")
        return set()

# ENV Variablen
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALLOWED_USERS = load_allowed_users()

# Validierung mit besseren Fehlermeldungen
if not TELEGRAM_TOKEN:
    raise RuntimeError("❌ TELEGRAM_TOKEN nicht gesetzt! Bitte in Railway konfigurieren.")
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY nicht gesetzt! Bitte in Railway konfigurieren.")
if not ALLOWED_USERS:
    logger.warning("⚠️ Keine User erlaubt - Bot wird alle Anfragen ablehnen!")

# ============================================================================
# CLIENTS INITIALISIEREN
# ============================================================================
try:
    bot = Bot(token=TELEGRAM_TOKEN)
    logger.info("✅ Telegram Bot initialisiert")
except Exception as e:
    logger.error(f"❌ Fehler bei Telegram Bot Init: {e}")
    raise

try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("✅ OpenAI Client initialisiert")
except Exception as e:
    logger.error(f"❌ Fehler bei OpenAI Client Init: {e}")
    raise

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(title="Telegram AI Bot", version="1.0.0")

# ============================================================================
# HELPER FUNKTIONEN
# ============================================================================
def is_rate_limited(user_id: int) -> bool:
    """Prüft ob User das Rate Limit erreicht hat."""
    now = time.time()
    timestamps = user_requests.get(user_id, [])
    
    # Entferne alte Timestamps
    timestamps = [t for t in timestamps if now - t < RATE_WINDOW]
    
    if len(timestamps) >= RATE_LIMIT:
        return True
    
    # Neuen Timestamp hinzufügen
    timestamps.append(now)
    user_requests[user_id] = timestamps
    return False

async def send_safe_message(chat_id: int, text: str) -> bool:
    """Sendet eine Telegram Nachricht mit Error Handling."""
    try:
        await bot.send_message(chat_id, text, parse_mode="HTML")
        return True
    except TelegramError as e:
        logger.error(f"❌ Telegram Fehler bei Chat {chat_id}: {e}")
        return False

def call_openai(user_message: str) -> str:
    """
    Ruft OpenAI API auf mit umfassendem Error Handling.
    
    Returns:
        str: LLM Response oder Fehlermeldung
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher, freundlicher KI-Assistent auf Deutsch."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        logger.info(f"✅ OpenAI Response generiert ({len(answer)} Zeichen)")
        return answer
        
    except RateLimitError as e:
        logger.error(f"❌ OpenAI Rate Limit: {e}")
        return (
            "⚠️ <b>OpenAI Quota überschritten</b>\n\n"
            "Der Bot ist vorübergehend nicht verfügbar. "
            "Bitte versuche es später noch einmal.\n\n"
            "💡 <i>Admin: Bitte OpenAI Guthaben aufladen.</i>"
        )
    
    except APIConnectionError as e:
        logger.error(f"❌ OpenAI Verbindungsfehler: {e}")
        return (
            "🔌 <b>Verbindungsfehler</b>\n\n"
            "Kann OpenAI nicht erreichen. Bitte versuche es später."
        )
    
    except APIError as e:
        logger.error(f"❌ OpenAI API Fehler: {e}")
        return (
            "⚙️ <b>API Fehler</b>\n\n"
            "Bei der Verarbeitung ist ein Fehler aufgetreten. "
            "Bitte versuche es erneut."
        )
    
    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler bei OpenAI: {e}", exc_info=True)
        return (
            "❌ <b>Unerwarteter Fehler</b>\n\n"
            "Etwas ist schiefgelaufen. Bitte kontaktiere den Admin."
        )

# ============================================================================
# WEBHOOK ENDPOINT
# ============================================================================
@app.post("/telegram")
async def telegram_webhook(req: Request):
    """
    Hauptendpoint für Telegram Webhook.
    
    Verarbeitet eingehende Nachrichten mit:
    - Whitelist-Überprüfung
    - Rate Limiting
    - Input Validierung
    - LLM Verarbeitung
    - Error Handling
    """
    try:
        data = await req.json()
        logger.debug(f"📩 Webhook erhalten: {json.dumps(data, indent=2)}")
        
        # Nur Nachrichten verarbeiten
        if "message" not in data:
            logger.debug("ℹ️ Kein 'message' Feld - ignoriere")
            return {"ok": True}
        
        message = data["message"]
        user_id = message.get("from", {}).get("id")
        chat_id = message.get("chat", {}).get("id")
        text = message.get("text", "")
        username = message.get("from", {}).get("username", "Unbekannt")
        
        # Validierung der wichtigsten Felder
        if not user_id or not chat_id:
            logger.warning("⚠️ Fehlende user_id oder chat_id")
            return {"ok": True}
        
        logger.info(f"👤 Nachricht von User {user_id} (@{username}): {text[:50]}...")
        
        # ========================================================================
        # 1️⃣ WHITELIST PRÜFUNG
        # ========================================================================
        if user_id not in ALLOWED_USERS:
            logger.warning(f"🚫 Unerlaubter Zugriff von User {user_id} (@{username})")
            await send_safe_message(
                chat_id,
                "⛔ <b>Zugriff verweigert</b>\n\n"
                "Du bist nicht berechtigt, diesen Bot zu nutzen.\n"
                f"Deine User ID: <code>{user_id}</code>"
            )
            return {"ok": True}
        
        # ========================================================================
        # 2️⃣ RATE LIMITING
        # ========================================================================
        if is_rate_limited(user_id):
            remaining = user_requests[user_id][0] + RATE_WINDOW - time.time()
            logger.warning(f"⏳ Rate Limit für User {user_id} - {int(remaining)}s verbleibend")
            await send_safe_message(
                chat_id,
                f"⏳ <b>Rate Limit erreicht</b>\n\n"
                f"Du hast das Limit von {RATE_LIMIT} Anfragen pro {RATE_WINDOW}s erreicht.\n"
                f"Bitte warte noch <b>{int(remaining)} Sekunden</b>."
            )
            return {"ok": True}
        
        # ========================================================================
        # 3️⃣ INPUT VALIDIERUNG
        # ========================================================================
        if not text or not text.strip():
            logger.info("ℹ️ Leere Nachricht - ignoriere")
            return {"ok": True}
        
        if len(text) > MAX_INPUT_LENGTH:
            logger.warning(f"✂️ Nachricht zu lang: {len(text)} Zeichen")
            await send_safe_message(
                chat_id,
                f"✂️ <b>Nachricht zu lang</b>\n\n"
                f"Maximale Länge: {MAX_INPUT_LENGTH} Zeichen\n"
                f"Deine Nachricht: {len(text)} Zeichen\n\n"
                f"Bitte kürze deine Nachricht."
            )
            return {"ok": True}
        
        # ========================================================================
        # 4️⃣ LLM VERARBEITUNG
        # ========================================================================
        logger.info(f"🤖 Verarbeite Anfrage von User {user_id}...")
        
        # Typing indicator (optional)
        try:
            await bot.send_chat_action(chat_id, "typing")
        except:
            pass  # Nicht kritisch wenn das fehlschlägt
        
        # OpenAI API Call
        ai_response = call_openai(text)
        
        # Antwort senden
        success = await send_safe_message(chat_id, ai_response)
        
        if success:
            logger.info(f"✅ Antwort an User {user_id} gesendet")
        else:
            logger.error(f"❌ Konnte Antwort nicht an User {user_id} senden")
        
        return {"ok": True}
    
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON Parse Fehler: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler im Webhook: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}

# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================
@app.get("/")
async def health_check():
    """Health Check Endpoint für Railway."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "allowed_users": len(ALLOWED_USERS),
        "rate_limit": f"{RATE_LIMIT}/{RATE_WINDOW}s"
    }

@app.get("/health")
async def detailed_health():
    """Detaillierter Health Check."""
    return {
        "status": "healthy",
        "telegram_configured": bool(TELEGRAM_TOKEN),
        "openai_configured": bool(OPENAI_API_KEY),
        "allowed_users_count": len(ALLOWED_USERS),
        "config": {
            "rate_limit": RATE_LIMIT,
            "rate_window": RATE_WINDOW,
            "max_input": MAX_INPUT_LENGTH,
            "max_tokens": MAX_TOKENS
        }
    }

# ============================================================================
# STARTUP EVENT
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """Wird beim Start der Anwendung ausgeführt."""
    logger.info("=" * 60)
    logger.info("🚀 TELEGRAM AI BOT GESTARTET")
    logger.info("=" * 60)
    logger.info(f"✅ Telegram Bot: Konfiguriert")
    logger.info(f"✅ OpenAI API: Konfiguriert")
    logger.info(f"✅ Erlaubte User: {len(ALLOWED_USERS)}")
    logger.info(f"✅ Rate Limit: {RATE_LIMIT}/{RATE_WINDOW}s")
    logger.info("=" * 60)
