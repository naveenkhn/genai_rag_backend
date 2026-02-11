import re

# Chit-chat keywords and detection function
CHITCHAT_KEYWORDS = ["hello", "hi", "hey", "how are you", "thanks", "thank you", "how are you ?"]

def is_chitchat(query: str) -> bool:
    q = query.lower().strip()
    for kw in CHITCHAT_KEYWORDS:
        if " " in kw:
            # Multi-word phrase: require exact match
            if q == kw:
                if 'logger' in globals() and logger:
                    logger.debug(f"[DEBUG] is_chitchat triggered. Query='{query}' matched phrase '{kw}'")
                return True
        else:
            # Single word: match whole word with word boundaries
            if re.search(rf"\b{re.escape(kw)}\b", q):
                if 'logger' in globals() and logger:
                    logger.debug(f"[DEBUG] is_chitchat triggered. Query='{query}' matched word '{kw}'")
                return True
    return False