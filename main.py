import os
import json
import logging
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse, JSONResponse

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trust-maker-bot")

META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MARGBOOKS_API_KEY = os.getenv("MARGBOOKS_API_KEY", "")
MARGBOOKS_API_URL = os.getenv("MARGBOOKS_API_URL", "https://api.margbooks.com/api/v1/item-stock")

GEMINI_MODEL = "gemini-2.5-flash"
GRAPH_API_VERSION = "v21.0"

GEMINI_SYSTEM_INSTRUCTION = (
    "You are the helpful sales executive for Trust Maker (bags and wallets). "
    "Provide crisp, polite answers for WhatsApp. If they ask to view products, "
    "tell them to type 'Menu' to view our interactive catalog."
)

app = FastAPI(title="Trust Maker WhatsApp Bot")

ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
http_client = httpx.AsyncClient(timeout=10.0)


@app.on_event("shutdown")
async def _shutdown() -> None:
    await http_client.aclose()


# ---------------------------------------------------------------------------
# Catalog definitions
# ---------------------------------------------------------------------------
# Each menu is a dict of header/body/button/section_title/rows.
# Rows are (id, title, description-or-None).

HOME_MENU = {
    "header": "Trust Maker",
    "body": "Welcome to Trust Maker! Please select an option from our collection:",
    "button": "Browse Store",
    "section_title": "Categories",
    "rows": [
        ("cat_wallet", "Wallets", "PU, Leather & Premium collection"),
        ("cat_bag", "Bags", "Coming Soon"),
        ("cat_best", "Best Sellers", "Top trending items"),
        ("cat_soon", "Coming Soon", "New arrivals preview"),
    ],
}

WALLET_MENU = {
    "header": "Wallet Collection",
    "body": "Select a wallet category to view styles:",
    "button": "Select Wallet Type",
    "section_title": "Wallet Types",
    "rows": [
        ("w_pu", "PU Wallet", None),
        ("w_leather", "Leather Wallet", None),
        ("w_fancy", "Fancy Wallet", None),
        ("w_card", "Card Holder", None),
        ("w_long", "Long Wallet", None),
        ("w_premium", "Premium Wallet", None),
    ],
}

PU_STYLE_MENU = {
    "header": "PU Wallet Styles",
    "body": "Choose a pattern / style:",
    "button": "Choose Style",
    "section_title": "Styles",
    "rows": [
        ("style_inner_leather", "Inner Leather", None),
        ("style_grain_printed", "Grain Printed", None),
        ("style_military", "Military", None),
    ],
}

# Menu-navigation ids -> which menu they open next.
MENU_TRIGGERS = {
    "cat_wallet": WALLET_MENU,
    "w_pu": PU_STYLE_MENU,
}

# ids that just show a "coming soon" style placeholder rather than a submenu
# or a real product (no further spec was given for these sections).
PLACEHOLDER_IDS = {
    "cat_bag": "Bags are coming soon! Type 'Menu' to go back to the main catalog.",
    "cat_soon": "New arrivals preview is coming soon! Type 'Menu' to go back.",
    "cat_best": "Our best sellers list is being updated. Type 'Menu' to browse categories.",
}

# Final, purchasable product ids -> the item name to look up in MargBooks.
PRODUCT_ITEMS = {
    "w_leather": "Leather Wallet",
    "w_fancy": "Fancy Wallet",
    "w_card": "Card Holder",
    "w_long": "Long Wallet",
    "w_premium": "Premium Wallet",
    "style_inner_leather": "PU Wallet - Inner Leather",
    "style_grain_printed": "PU Wallet - Grain Printed",
    "style_military": "PU Wallet - Military",
}


# ---------------------------------------------------------------------------
# 1. Webhook handshake
# ---------------------------------------------------------------------------
@app.get("/webhook")
async def verify_webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == META_VERIFY_TOKEN:
        logger.info("Webhook verified successfully.")
        return PlainTextResponse(content=challenge or "", status_code=200)

    logger.warning("Webhook verification failed (mode=%s).", mode)
    return Response(status_code=403)


# ---------------------------------------------------------------------------
# 2. Incoming message handler
# ---------------------------------------------------------------------------
@app.post("/webhook")
async def receive_message(request: Request):
    try:
        payload = await request.json()
    except Exception:
        logger.warning("Received non-JSON or empty payload on webhook.")
        return JSONResponse({"status": "ok"})

    parsed = _extract_message(payload)
    if parsed is None:
        # Status receipts (sent/delivered/read) or anything with no
        # actionable message land here -- ack immediately.
        return JSONResponse({"status": "ok"})

    from_number, msg_type, value = parsed

    try:
        if msg_type == "list_reply":
            await handle_list_selection(from_number, value)
        else:  # msg_type == "text"
            await handle_text_message(from_number, value)
    except Exception:
        # Never let an internal failure surface to Meta -- always 200 so it
        # doesn't endlessly retry the same webhook delivery.
        logger.exception("Unhandled error while processing incoming message.")

    return JSONResponse({"status": "ok"})


def _extract_message(payload: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    """Safely pulls (from_number, msg_type, value) out of a WhatsApp webhook
    payload, where msg_type is "text" or "list_reply" and value is the
    message body or the selected row id respectively.
    Returns None for status updates, malformed/empty payloads, or message
    types we don't act on (images, audio, button replies, etc.)."""
    try:
        entry = payload.get("entry", [])[0]
        change = entry.get("changes", [])[0]
        value = change.get("value", {})

        messages = value.get("messages")
        if not messages:
            # Typically a `statuses` (sent/delivered/read) callback -- ignore.
            return None

        message = messages[0]
        from_number = message.get("from")
        if not from_number:
            return None

        msg_type = message.get("type")

        if msg_type == "text":
            body = message.get("text", {}).get("body")
            if not body:
                return None
            return from_number, "text", body

        if msg_type == "interactive":
            interactive = message.get("interactive", {})
            if interactive.get("type") == "list_reply":
                row_id = interactive.get("list_reply", {}).get("id")
                if not row_id:
                    return None
                return from_number, "list_reply", row_id
            # button_reply or other interactive subtypes: not used by this flow
            return None

        # Images, audio, location, etc. -- not handled.
        return None
    except (KeyError, IndexError, AttributeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Message routing
# ---------------------------------------------------------------------------
async def handle_text_message(from_number: str, text: str) -> None:
    if text.strip().lower() in ("menu", "hi", "hello", "start", "hey"):
        await send_list_message(from_number, HOME_MENU)
        return

    reply = await generate_gemini_reply(text)
    await send_text_message(from_number, reply)


async def handle_list_selection(from_number: str, row_id: str) -> None:
    if row_id in MENU_TRIGGERS:
        await send_list_message(from_number, MENU_TRIGGERS[row_id])
        return

    if row_id in PLACEHOLDER_IDS:
        await send_text_message(from_number, PLACEHOLDER_IDS[row_id])
        return

    if row_id in PRODUCT_ITEMS:
        item_name = PRODUCT_ITEMS[row_id]
        reply = await build_product_reply(item_name)
        await send_text_message(from_number, reply)
        return

    # Unknown/unmapped id -- fall back to the home menu.
    logger.warning("Unrecognized list selection id: %s", row_id)
    await send_text_message(from_number, "Sorry, I didn't recognize that option.")
    await send_list_message(from_number, HOME_MENU)


# ---------------------------------------------------------------------------
# 4. MargBooks stock lookup & inventory confirmation
# ---------------------------------------------------------------------------
async def fetch_margbooks_items() -> Optional[List[Dict[str, Any]]]:
    """Fetches the raw MargBooks stock list. Returns None (never raises) if
    the API key is missing or the call fails/times out for any reason."""
    if not MARGBOOKS_API_KEY:
        logger.info("MARGBOOKS_API_KEY not configured; skipping live stock lookup.")
        return None

    headers = {"Authorization": f"Bearer {MARGBOOKS_API_KEY}"}
    try:
        response = await http_client.get(MARGBOOKS_API_URL, headers=headers, timeout=8.0)
        response.raise_for_status()
        data = response.json()
    except (httpx.TimeoutException, httpx.HTTPError, ValueError) as exc:
        logger.error("MargBooks inventory fetch failed: %s", exc)
        return None

    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        logger.error("Unexpected MargBooks response shape: %r", data)
        return None
    return items


def _find_item(items: List[Dict[str, Any]], item_name: str) -> Optional[Dict[str, Any]]:
    target = item_name.strip().lower()
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("item_name") or "").strip().lower()
        if not name:
            continue
        if name == target or target in name or name in target:
            return item
    return None


async def build_product_reply(item_name: str) -> str:
    """Looks up `item_name` in MargBooks and formats a WhatsApp-ready reply
    covering all three required cases: not configured/unreachable, found but
    out of stock, and found with stock available."""
    items = await fetch_margbooks_items()

    if items is None:
        return (
            f"*{item_name}*\n"
            "Live stock check is temporarily unavailable, but this item is part "
            "of our regular collection. Please message us to confirm current "
            "availability and pricing."
        )

    item = _find_item(items, item_name)
    if item is None:
        return (
            f"*{item_name}*\n"
            "We couldn't find live stock details for this item right now. "
            "Please message us to confirm availability and pricing."
        )

    try:
        raw_qty = item.get("quantity", item.get("stock", 0))
        qty = float(raw_qty) if raw_qty is not None else 0.0
    except (TypeError, ValueError):
        qty = 0.0

    rate = item.get("rate") or item.get("price") or "N/A"

    if qty <= 0:
        return (
            f"*{item_name}* is currently out of stock.\n"
            "Would you like to see other styles? Type 'Menu' to browse our catalog."
        )

    return f"*{item_name}*\nAvailable quantity: {qty}\nRate: {rate}"


# ---------------------------------------------------------------------------
# 5. Google Gemini freeform assistant
# ---------------------------------------------------------------------------
async def generate_gemini_reply(customer_message: str) -> str:
    if ai_client is None:
        logger.error("GEMINI_API_KEY is not configured; cannot generate a reply.")
        return "Sorry, our assistant is temporarily unavailable. Please try again shortly."

    def _call_gemini() -> str:
        response = ai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=customer_message,
            config=types.GenerateContentConfig(system_instruction=GEMINI_SYSTEM_INSTRUCTION),
        )
        return (response.text or "").strip()

    try:
        # The genai SDK's client is synchronous; run it in a worker thread so
        # it doesn't block the event loop.
        text = await asyncio.to_thread(_call_gemini)
    except Exception:
        logger.exception("Gemini generation failed.")
        return "Sorry, I couldn't process that just now. Please try again in a moment."

    return text or "Sorry, I didn't quite get that -- could you rephrase?"


# ---------------------------------------------------------------------------
# 6. Outgoing WhatsApp messages (text + interactive list)
# ---------------------------------------------------------------------------
async def _post_to_whatsapp(payload: Dict[str, Any]) -> None:
    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    try:
        response = await http_client.post(url, headers=headers, json=payload, timeout=8.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("Failed to send WhatsApp message to %s: %s", payload.get("to"), exc)


async def send_text_message(to_number: str, text: str) -> None:
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": text[:4096]},  # WhatsApp text bodies cap at 4096 chars
    }
    await _post_to_whatsapp(payload)


async def send_list_message(to_number: str, menu: Dict[str, Any]) -> None:
    rows = [
        {"id": row_id, "title": title, **({"description": desc} if desc else {})}
        for row_id, title, desc in menu["rows"]
    ]
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "header": {"type": "text", "text": menu["header"]},
            "body": {"text": menu["body"]},
            "action": {
                "button": menu["button"],
                "sections": [{"title": menu["section_title"], "rows": rows}],
            },
        },
    }
    await _post_to_whatsapp(payload)


@app.get("/")
async def root():
    return {"status": "Trust Maker WhatsApp bot is running."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
