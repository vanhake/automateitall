import os
import logging
from fastapi import FastAPI, Request, HTTPException
from telegram import Bot, InputFile
from telegram.error import TelegramError
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
import time
import json
from typing import Dict, List, Set, Optional
from datetime import datetime
import httpx
from io import BytesIO
import requests
import re

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

# Separate Rate Limits für Bilder (teurer!)
IMAGE_RATE_LIMIT = 5   # max. Bildanfragen
IMAGE_RATE_WINDOW = 300  # pro 5 Minuten

user_requests: Dict[int, List[float]] = {}
user_image_requests: Dict[int, List[float]] = {}

# ============================================================================
# GRUPPENCHAT KONFIGURATION
# ============================================================================
# Bot muss mit @botname oder /befehl@botname angesprochen werden
REQUIRE_MENTION_IN_GROUPS = True  # True = nur reagieren wenn erwähnt, False = auf alle Nachrichten reagieren

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

def load_allowed_groups() -> Set[int]:
    """Lädt erlaubte Gruppen-IDs aus Umgebungsvariable."""
    raw = os.getenv("ALLOWED_GROUPS", "")
    if not raw:
        logger.info("ℹ️ ALLOWED_GROUPS nicht gesetzt - Gruppenchats deaktiviert")
        return set()
    
    try:
        groups = {int(gid.strip()) for gid in raw.split(",") if gid.strip()}
        logger.info(f"✅ {len(groups)} erlaubte Gruppen geladen: {groups}")
        return groups
    except ValueError as e:
        logger.error(f"❌ Fehler beim Parsen von ALLOWED_GROUPS: {e}")
        return set()

# ENV Variablen
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALLOWED_USERS = load_allowed_users()
ALLOWED_GROUPS = load_allowed_groups()

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
    # Bot Username für Mentions abrufen
    import asyncio
    BOT_USERNAME = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bot_info = loop.run_until_complete(bot.get_me())
        BOT_USERNAME = bot_info.username
        logger.info(f"✅ Bot Username: @{BOT_USERNAME}")
        loop.close()
    except Exception as e:
        logger.warning(f"⚠️ Konnte Bot Username nicht abrufen: {e}")
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
app = FastAPI(title="Telegram AI Bot with Image Generation & Group Support", version="2.1.0")

# ============================================================================
# HELPER FUNKTIONEN
# ============================================================================
def is_group_chat(chat_type: str) -> bool:
    """Prüft ob es sich um einen Gruppenchat handelt."""
    return chat_type in ["group", "supergroup"]

def is_bot_mentioned(text: str, entities: list, bot_username: str) -> bool:
    """
    Prüft ob der Bot in der Nachricht erwähnt wurde.
    
    Berücksichtigt:
    - @botname mentions
    - /command@botname
    - Direct replies (wird separat geprüft)
    """
    if not text or not bot_username:
        return False
    
    # Prüfe auf @botname
    mention_pattern = f"@{bot_username}"
    if mention_pattern.lower() in text.lower():
        return True
    
    # Prüfe entities für mentions
    if entities:
        for entity in entities:
            if entity.get("type") == "mention":
                offset = entity.get("offset", 0)
                length = entity.get("length", 0)
                mentioned = text[offset:offset+length]
                if mentioned.lower() == mention_pattern.lower():
                    return True
            
            # Bot commands mit @botname
            if entity.get("type") == "bot_command":
                offset = entity.get("offset", 0)
                length = entity.get("length", 0)
                command = text[offset:offset+length]
                if mention_pattern.lower() in command.lower():
                    return True
    
    return False

def remove_bot_mention(text: str, bot_username: str) -> str:
    """Entfernt Bot-Mentions aus dem Text."""
    if not text or not bot_username:
        return text
    
    # Entferne @botname
    text = re.sub(f"@{bot_username}", "", text, flags=re.IGNORECASE)
    
    # Entferne /command@botname und mache es zu /command
    text = re.sub(f"(/\w+)@{bot_username}", r"\1", text, flags=re.IGNORECASE)
    
    return text.strip()

def is_rate_limited(user_id: int) -> bool:
    """Prüft ob User das Text Rate Limit erreicht hat."""
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

def is_image_rate_limited(user_id: int) -> tuple[bool, Optional[int]]:
    """
    Prüft ob User das Bild Rate Limit erreicht hat.
    
    Returns:
        (is_limited, remaining_seconds)
    """
    now = time.time()
    timestamps = user_image_requests.get(user_id, [])
    
    # Entferne alte Timestamps
    timestamps = [t for t in timestamps if now - t < IMAGE_RATE_WINDOW]
    
    if len(timestamps) >= IMAGE_RATE_LIMIT:
        remaining = int(timestamps[0] + IMAGE_RATE_WINDOW - now)
        return True, remaining
    
    # Neuen Timestamp hinzufügen
    timestamps.append(now)
    user_image_requests[user_id] = timestamps
    return False, None

async def send_safe_message(chat_id: int, text: str, reply_to_message_id: Optional[int] = None) -> bool:
    """Sendet eine Telegram Nachricht mit Error Handling."""
    try:
        await bot.send_message(
            chat_id, 
            text, 
            parse_mode="HTML",
            reply_to_message_id=reply_to_message_id
        )
        return True
    except TelegramError as e:
        logger.error(f"❌ Telegram Fehler bei Chat {chat_id}: {e}")
        return False

async def send_photo(chat_id: int, photo_bytes: bytes, caption: str = "", reply_to_message_id: Optional[int] = None) -> bool:
    """Sendet ein Foto an den User."""
    try:
        await bot.send_photo(
            chat_id=chat_id,
            photo=InputFile(BytesIO(photo_bytes), filename="image.png"),
            caption=caption,
            parse_mode="HTML",
            reply_to_message_id=reply_to_message_id
        )
        return True
    except TelegramError as e:
        logger.error(f"❌ Fehler beim Senden des Fotos: {e}")
        return False

def parse_image_command(text: str) -> tuple[str, str]:
    """
    Parst Bildkommandos.
    
    Returns:
        (command, prompt)
    
    Beispiele:
        "/bild ein roter Apfel" -> ("generate", "ein roter Apfel")
        "/edit mach den Himmel blau" -> ("edit", "mach den Himmel blau")
    """
    text = text.strip()
    
    if text.startswith("/bild ") or text.startswith("/generate "):
        parts = text.split(" ", 1)
        return ("generate", parts[1] if len(parts) > 1 else "")
    
    if text.startswith("/edit ") or text.startswith("/bearbeite "):
        parts = text.split(" ", 1)
        return ("edit", parts[1] if len(parts) > 1 else "")
    
    if text.startswith("/variation") or text.startswith("/variante"):
        return ("variation", "")
    
    return ("text", text)

def generate_image(prompt: str, size: str = "1024x1024", quality: str = "standard") -> tuple[Optional[bytes], Optional[str]]:
    """
    Generiert ein Bild mit DALL-E 3.
    
    Args:
        prompt: Beschreibung des zu generierenden Bildes
        size: Bildgröße (1024x1024, 1792x1024, 1024x1792)
        quality: Qualität (standard, hd)
    
    Returns:
        (image_bytes, error_message)
    """
    try:
        logger.info(f"🎨 Generiere Bild mit Prompt: {prompt[:50]}...")
        
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            n=1
        )
        
        image_url = response.data[0].url
        revised_prompt = response.data[0].revised_prompt
        
        logger.info(f"✅ Bild generiert. Revised Prompt: {revised_prompt[:50]}...")
        
        # Bild herunterladen
        img_response = requests.get(image_url, timeout=30)
        img_response.raise_for_status()
        
        return img_response.content, None
        
    except RateLimitError as e:
        logger.error(f"❌ OpenAI Rate Limit bei Bildgenerierung: {e}")
        return None, (
            "⚠️ <b>OpenAI Quota überschritten</b>\n\n"
            "Bildgenerierung vorübergehend nicht verfügbar."
        )
    
    except APIError as e:
        logger.error(f"❌ OpenAI API Fehler: {e}")
        
        # Prüfe auf Content Policy Violation
        if "content_policy_violation" in str(e).lower():
            return None, (
                "🚫 <b>Content Policy Verstoß</b>\n\n"
                "Dein Prompt verstößt gegen die OpenAI Content Policy. "
                "Bitte formuliere deine Anfrage anders."
            )
        
        return None, (
            "⚙️ <b>API Fehler</b>\n\n"
            "Bei der Bildgenerierung ist ein Fehler aufgetreten."
        )
    
    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler bei Bildgenerierung: {e}", exc_info=True)
        return None, (
            "❌ <b>Unerwarteter Fehler</b>\n\n"
            "Bildgenerierung fehlgeschlagen. Bitte versuche es erneut."
        )

async def download_telegram_photo(file_id: str) -> Optional[bytes]:
    """Lädt ein Foto von Telegram herunter."""
    try:
        file = await bot.get_file(file_id)
        file_bytes = await file.download_as_bytearray()
        return bytes(file_bytes)
    except Exception as e:
        logger.error(f"❌ Fehler beim Download des Telegram-Fotos: {e}")
        return None

def edit_image(image_bytes: bytes, prompt: str, size: str = "1024x1024") -> tuple[Optional[bytes], Optional[str]]:
    """
    Bearbeitet ein Bild mit DALL-E 2.
    
    Args:
        image_bytes: Original-Bild als Bytes
        prompt: Beschreibung der gewünschten Änderungen
        size: Bildgröße (256x256, 512x512, 1024x1024)
    
    Returns:
        (edited_image_bytes, error_message)
    """
    try:
        logger.info(f"✏️ Bearbeite Bild mit Prompt: {prompt[:50]}...")
        
        # DALL-E 2 für Image Editing
        response = client.images.edit(
            model="dall-e-2",
            image=image_bytes,
            prompt=prompt,
            size=size,
            n=1
        )
        
        image_url = response.data[0].url
        
        logger.info(f"✅ Bild bearbeitet")
        
        # Bild herunterladen
        img_response = requests.get(image_url, timeout=30)
        img_response.raise_for_status()
        
        return img_response.content, None
        
    except RateLimitError as e:
        logger.error(f"❌ OpenAI Rate Limit bei Bildbearbeitung: {e}")
        return None, "⚠️ OpenAI Quota überschritten"
    
    except Exception as e:
        logger.error(f"❌ Fehler bei Bildbearbeitung: {e}", exc_info=True)
        return None, f"❌ Bildbearbeitung fehlgeschlagen: {str(e)}"

def create_image_variation(image_bytes: bytes, size: str = "1024x1024") -> tuple[Optional[bytes], Optional[str]]:
    """
    Erstellt eine Variation eines Bildes mit DALL-E 2.
    
    Args:
        image_bytes: Original-Bild als Bytes
        size: Bildgröße (256x256, 512x512, 1024x1024)
    
    Returns:
        (variation_image_bytes, error_message)
    """
    try:
        logger.info(f"🔄 Erstelle Bildvariation...")
        
        response = client.images.create_variation(
            model="dall-e-2",
            image=image_bytes,
            size=size,
            n=1
        )
        
        image_url = response.data[0].url
        
        logger.info(f"✅ Variation erstellt")
        
        # Bild herunterladen
        img_response = requests.get(image_url, timeout=30)
        img_response.raise_for_status()
        
        return img_response.content, None
        
    except Exception as e:
        logger.error(f"❌ Fehler bei Bildvariation: {e}", exc_info=True)
        return None, f"❌ Variation fehlgeschlagen: {str(e)}"

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
    
    Verarbeitet:
    - Private Chats
    - Gruppenchats (mit @mention oder reply)
    - Text-Nachrichten (Chat)
    - Bildgenerierung (/bild)
    - Bildbearbeitung (/edit + Foto)
    - Bildvariationen (/variation + Foto)
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
        chat_type = message.get("chat", {}).get("type")
        message_id = message.get("message_id")
        text = message.get("text", "")
        caption = message.get("caption", "")
        username = message.get("from", {}).get("username", "Unbekannt")
        entities = message.get("entities", [])
        
        # Reply to message?
        reply_to_message = message.get("reply_to_message")
        is_reply_to_bot = False
        if reply_to_message:
            replied_user = reply_to_message.get("from", {}).get("id")
            if replied_user == (await bot.get_me()).id:
                is_reply_to_bot = True
        
        # Photo vorhanden?
        has_photo = "photo" in message
        photo_file_id = message.get("photo", [{}])[-1].get("file_id") if has_photo else None
        
        # Wenn Foto mit Caption, nutze Caption als Text
        if has_photo and caption:
            text = caption
            entities = message.get("caption_entities", [])
        
        # Validierung der wichtigsten Felder
        if not user_id or not chat_id:
            logger.warning("⚠️ Fehlende user_id oder chat_id")
            return {"ok": True}
        
        logger.info(f"👤 Nachricht von User {user_id} (@{username}) in {chat_type}: {text[:50]}...")
        
        # ========================================================================
        # GRUPPENCHAT PRÜFUNG
        # ========================================================================
        is_group = is_group_chat(chat_type)
        
        if is_group:
            # Prüfe ob Gruppe erlaubt ist
            if ALLOWED_GROUPS and chat_id not in ALLOWED_GROUPS:
                logger.warning(f"🚫 Nachricht aus nicht-erlaubter Gruppe {chat_id}")
                return {"ok": True}
            
            # In Gruppen: Nur reagieren wenn Bot erwähnt wird oder Reply an Bot
            if REQUIRE_MENTION_IN_GROUPS:
                bot_mentioned = is_bot_mentioned(text, entities, BOT_USERNAME)
                
                if not bot_mentioned and not is_reply_to_bot:
                    logger.debug(f"ℹ️ Bot nicht erwähnt in Gruppe {chat_id} - ignoriere")
                    return {"ok": True}
                
                # Entferne Bot-Mention aus Text für saubere Verarbeitung
                if bot_mentioned:
                    text = remove_bot_mention(text, BOT_USERNAME)
                    logger.info(f"✅ Bot erwähnt in Gruppe - Text: {text[:50]}...")
        
        # ========================================================================
        # WHITELIST PRÜFUNG (nur für User, nicht für Gruppen)
        # ========================================================================
        if not is_group and user_id not in ALLOWED_USERS:
            logger.warning(f"🚫 Unerlaubter Zugriff von User {user_id} (@{username})")
            await send_safe_message(
                chat_id,
                "⛔ <b>Zugriff verweigert</b>\n\n"
                "Du bist nicht berechtigt, diesen Bot zu nutzen.\n"
                f"Deine User ID: <code>{user_id}</code>",
                reply_to_message_id=message_id if is_group else None
            )
            return {"ok": True}
        
        # In Gruppen: Prüfe ob User in ALLOWED_USERS (falls gesetzt)
        if is_group and ALLOWED_USERS and user_id not in ALLOWED_USERS:
            logger.warning(f"🚫 User {user_id} nicht in ALLOWED_USERS - ignoriere in Gruppe")
            return {"ok": True}
        
        # ========================================================================
        # KOMMANDO PARSING
        # ========================================================================
        command, prompt = parse_image_command(text)
        
        # Help Command
        if text.strip() in ["/help", "/hilfe", "/start"]:
            help_text = (
                "🤖 <b>AI Bot - Hilfe</b>\n\n"
                "<b>💬 Chat:</b>\n"
            )
            
            if is_group:
                help_text += (
                    f"Erwähne mich mit @{BOT_USERNAME} oder antworte auf meine Nachrichten.\n"
                    f"Beispiel: <code>@{BOT_USERNAME} Wie geht es dir?</code>\n\n"
                )
            else:
                help_text += "Schreibe einfach eine Nachricht für normalen Chat.\n\n"
            
            help_text += (
                "<b>🎨 Bildgenerierung:</b>\n"
                "<code>/bild [Beschreibung]</code>\n"
                "Beispiel: <code>/bild ein süßer Hund im Park</code>\n\n"
                "<b>✏️ Bildbearbeitung:</b>\n"
                "1. Sende ein Foto mit Caption: <code>/edit [Änderung]</code>\n"
                "Beispiel: Foto + Caption <code>/edit mach den Himmel orange</code>\n\n"
                "<b>🔄 Bildvariation:</b>\n"
                "Sende ein Foto mit Caption: <code>/variation</code>\n\n"
                f"<b>⏱ Limits:</b>\n"
                f"• Text: {RATE_LIMIT} Anfragen / {RATE_WINDOW}s\n"
                f"• Bilder: {IMAGE_RATE_LIMIT} Anfragen / {IMAGE_RATE_WINDOW//60} Minuten"
            )
            
            if is_group:
                help_text += f"\n\n💡 <i>In Gruppen muss der Bot mit @{BOT_USERNAME} erwähnt werden.</i>"
            
            await send_safe_message(chat_id, help_text, reply_to_message_id=message_id if is_group else None)
            return {"ok": True}
        
        # ========================================================================
        # BILDKOMMANDOS
        # ========================================================================
        if command in ["generate", "edit", "variation"]:
            # Image Rate Limit Check
            is_limited, remaining = is_image_rate_limited(user_id)
            if is_limited:
                logger.warning(f"🎨 Bild Rate Limit für User {user_id} - {remaining}s verbleibend")
                await send_safe_message(
                    chat_id,
                    f"🎨 <b>Bild Rate Limit erreicht</b>\n\n"
                    f"Du hast das Limit von {IMAGE_RATE_LIMIT} Bildanfragen pro {IMAGE_RATE_WINDOW//60} Minuten erreicht.\n"
                    f"Bitte warte noch <b>{remaining} Sekunden</b>.\n\n"
                    f"💡 <i>Bildgenerierung ist teurer als Text-Chat.</i>",
                    reply_to_message_id=message_id if is_group else None
                )
                return {"ok": True}
            
            # === BILDGENERIERUNG ===
            if command == "generate":
                if not prompt:
                    await send_safe_message(
                        chat_id,
                        "❌ <b>Fehlender Prompt</b>\n\n"
                        "Bitte beschreibe das Bild, das du generieren möchtest.\n\n"
                        "Beispiel: <code>/bild ein roter Sportwagen vor einem Sonnenuntergang</code>",
                        reply_to_message_id=message_id if is_group else None
                    )
                    return {"ok": True}
                
                await send_safe_message(
                    chat_id, 
                    "🎨 Generiere Bild... Dies kann 10-30 Sekunden dauern.",
                    reply_to_message_id=message_id if is_group else None
                )
                await bot.send_chat_action(chat_id, "upload_photo")
                
                image_bytes, error = generate_image(prompt)
                
                if error:
                    await send_safe_message(chat_id, error, reply_to_message_id=message_id if is_group else None)
                else:
                    success = await send_photo(
                        chat_id,
                        image_bytes,
                        caption=f"🎨 <b>Generiert</b>\n<i>Prompt: {prompt[:100]}...</i>",
                        reply_to_message_id=message_id if is_group else None
                    )
                    if success:
                        logger.info(f"✅ Bild an User {user_id} gesendet")
                
                return {"ok": True}
            
            # === BILDBEARBEITUNG ===
            elif command == "edit":
                if not has_photo:
                    await send_safe_message(
                        chat_id,
                        "📸 <b>Kein Foto gefunden</b>\n\n"
                        "Bitte sende ein Foto mit Caption:\n"
                        "<code>/edit [Beschreibung der Änderung]</code>\n\n"
                        "Beispiel: Sende Foto mit Caption <code>/edit mach den Hintergrund zu einem Strand</code>",
                        reply_to_message_id=message_id if is_group else None
                    )
                    return {"ok": True}
                
                if not prompt:
                    await send_safe_message(
                        chat_id,
                        "❌ <b>Fehlende Beschreibung</b>\n\n"
                        "Bitte beschreibe die gewünschte Änderung.\n\n"
                        "Beispiel: Sende Foto mit Caption <code>/edit mach es schwarz-weiß</code>",
                        reply_to_message_id=message_id if is_group else None
                    )
                    return {"ok": True}
                
                await send_safe_message(
                    chat_id, 
                    "✏️ Bearbeite Bild... Dies kann 10-20 Sekunden dauern.",
                    reply_to_message_id=message_id if is_group else None
                )
                await bot.send_chat_action(chat_id, "upload_photo")
                
                # Foto herunterladen
                original_image = await download_telegram_photo(photo_file_id)
                if not original_image:
                    await send_safe_message(
                        chat_id, 
                        "❌ Konnte Foto nicht herunterladen.",
                        reply_to_message_id=message_id if is_group else None
                    )
                    return {"ok": True}
                
                edited_image, error = edit_image(original_image, prompt)
                
                if error:
                    await send_safe_message(chat_id, error, reply_to_message_id=message_id if is_group else None)
                else:
                    await send_photo(
                        chat_id,
                        edited_image,
                        caption=f"✏️ <b>Bearbeitet</b>\n<i>Änderung: {prompt[:100]}...</i>",
                        reply_to_message_id=message_id if is_group else None
                    )
                
                return {"ok": True}
            
            # === BILDVARIATION ===
            elif command == "variation":
                if not has_photo:
                    await send_safe_message(
                        chat_id,
                        "📸 <b>Kein Foto gefunden</b>\n\n"
                        "Bitte sende ein Foto mit Caption:\n"
                        "<code>/variation</code>",
                        reply_to_message_id=message_id if is_group else None
                    )
                    return {"ok": True}
                
                await send_safe_message(
                    chat_id, 
                    "🔄 Erstelle Variation... Dies kann 10-20 Sekunden dauern.",
                    reply_to_message_id=message_id if is_group else None
                )
                await bot.send_chat_action(chat_id, "upload_photo")
                
                # Foto herunterladen
                original_image = await download_telegram_photo(photo_file_id)
                if not original_image:
                    await send_safe_message(
                        chat_id, 
                        "❌ Konnte Foto nicht herunterladen.",
                        reply_to_message_id=message_id if is_group else None
                    )
                    return {"ok": True}
                
                variation_image, error = create_image_variation(original_image)
                
                if error:
                    await send_safe_message(chat_id, error, reply_to_message_id=message_id if is_group else None)
                else:
                    await send_photo(
                        chat_id,
                        variation_image,
                        caption="🔄 <b>Variation erstellt</b>",
                        reply_to_message_id=message_id if is_group else None
                    )
                
                return {"ok": True}
        
        # ========================================================================
        # NORMALER TEXT-CHAT
        # ========================================================================
        
        # Text Rate Limit
        if is_rate_limited(user_id):
            remaining = user_requests[user_id][0] + RATE_WINDOW - time.time()
            logger.warning(f"⏳ Rate Limit für User {user_id} - {int(remaining)}s verbleibend")
            await send_safe_message(
                chat_id,
                f"⏳ <b>Rate Limit erreicht</b>\n\n"
                f"Du hast das Limit von {RATE_LIMIT} Anfragen pro {RATE_WINDOW}s erreicht.\n"
                f"Bitte warte noch <b>{int(remaining)} Sekunden</b>.",
                reply_to_message_id=message_id if is_group else None
            )
            return {"ok": True}
        
        # Input Validierung
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
                f"Bitte kürze deine Nachricht.",
                reply_to_message_id=message_id if is_group else None
            )
            return {"ok": True}
        
        # LLM Verarbeitung
        logger.info(f"🤖 Verarbeite Anfrage von User {user_id}...")
        
        try:
            await bot.send_chat_action(chat_id, "typing")
        except:
            pass
        
        ai_response = call_openai(text)
        
        success = await send_safe_message(
            chat_id, 
            ai_response,
            reply_to_message_id=message_id if is_group else None
        )
        
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
# HEALTH CHECK ENDPOINTS
# ============================================================================
@app.get("/")
async def health_check():
    """Health Check Endpoint für Railway."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "bot_username": BOT_USERNAME,
        "allowed_users": len(ALLOWED_USERS),
        "allowed_groups": len(ALLOWED_GROUPS),
        "rate_limit_text": f"{RATE_LIMIT}/{RATE_WINDOW}s",
        "rate_limit_images": f"{IMAGE_RATE_LIMIT}/{IMAGE_RATE_WINDOW//60}min",
        "features": ["text_chat", "image_generation", "image_editing", "image_variation", "group_chat"]
    }

@app.get("/health")
async def detailed_health():
    """Detaillierter Health Check."""
    return {
        "status": "healthy",
        "telegram_configured": bool(TELEGRAM_TOKEN),
        "openai_configured": bool(OPENAI_API_KEY),
        "bot_username": BOT_USERNAME,
        "allowed_users_count": len(ALLOWED_USERS),
        "allowed_groups_count": len(ALLOWED_GROUPS),
        "group_mention_required": REQUIRE_MENTION_IN_GROUPS,
        "config": {
            "rate_limit_text": RATE_LIMIT,
            "rate_window_text": RATE_WINDOW,
            "rate_limit_images": IMAGE_RATE_LIMIT,
            "rate_window_images": IMAGE_RATE_WINDOW,
            "max_input": MAX_INPUT_LENGTH,
            "max_tokens": MAX_TOKENS
        },
        "features": {
            "text_chat": True,
            "image_generation": True,
            "image_editing": True,
            "image_variation": True,
            "group_chat": True,
            "mention_detection": True
        }
    }

# ============================================================================
# STARTUP EVENT
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """Wird beim Start der Anwendung ausgeführt."""
    logger.info("=" * 60)
    logger.info("🚀 TELEGRAM AI BOT MIT GRUPPEN-SUPPORT GESTARTET")
    logger.info("=" * 60)
    logger.info(f"✅ Telegram Bot: Konfiguriert (@{BOT_USERNAME})")
    logger.info(f"✅ OpenAI API: Konfiguriert")
    logger.info(f"✅ Erlaubte User: {len(ALLOWED_USERS)}")
    logger.info(f"✅ Erlaubte Gruppen: {len(ALLOWED_GROUPS)}")
    logger.info(f"✅ Gruppen-Modus: {'@mention erforderlich' if REQUIRE_MENTION_IN_GROUPS else 'Alle Nachrichten'}")
    logger.info(f"✅ Text Rate Limit: {RATE_LIMIT}/{RATE_WINDOW}s")
    logger.info(f"✅ Bild Rate Limit: {IMAGE_RATE_LIMIT}/{IMAGE_RATE_WINDOW}s")
    logger.info(f"✅ Features: Text, Bilder, Gruppen")
    logger.info("=" * 60)
