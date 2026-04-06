"""
Send Telegram Notification - ComfyUI Node
=========================================
Sends a notification to telegram once a job is done.

┌─────────────────────────────────────┐
│ Send Telegram Notification          │
├─────────────────────────────────────┤
│ ○  Any                              │
│ <→ TEXT INPUT_message text>         │
│ - Auto Populated to:                │
│   %HMSf% is finished                │
│                                     │
└─────────────────────────────────────┘
"""
import requests
from datetime import datetime
from ._fg_helperfunctions import log
from .. import TELEGRAM_PRIVATE_API, TELEGRAM_CHAT_ID

class SendTelegramNotification:
    def __init__(self):
        self.NODE_NAME = "Send Telegram Notification"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": ("*", {}),
                "message_text": ("STRING", {
                    "default": "Job finished at %HMSf%",
                    "multiline": True,
                }),
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "execute"
    CATEGORY = "Farrenzo's Garbage/Utils"
    OUTPUT_NODE = True  # Important: marks this as an endpoint node
    
    # Passthrough the input type
    @classmethod
    def VALIDATE_INPUTS(cls, anything=None, **kwargs):
        return True

    def execute(self, anything, message_text: str, enabled: bool):
        if not enabled:
            return (anything,)
        
        # Replace time placeholders
        now = datetime.now()
        formatted_message = message_text.replace("%HMSf%", now.strftime("%H:%M:%S.%f")[:-3])
        formatted_message = formatted_message.replace("%HMS%", now.strftime("%H:%M:%S"))
        formatted_message = formatted_message.replace("%DATE%", now.strftime("%Y-%m-%d"))
        
        self.send_telegram_notification(formatted_message)
        
        return (anything,)

    def send_telegram_notification(self, message_text: str):
        """Sends a text message to a specified Telegram chat."""
        print(TELEGRAM_PRIVATE_API)
        print(TELEGRAM_CHAT_ID)
        if not TELEGRAM_PRIVATE_API or not TELEGRAM_CHAT_ID:
            log(f"{self.NODE_NAME}: Error sending notification. No credentials set up.")
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_PRIVATE_API}/sendMessage"

        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message_text,
            "parse_mode": "Markdown"
        }
        try:
            response = requests.post(url, data=payload)
            response.raise_for_status()
            log(f"{self.NODE_NAME}:Notification sent: {message_text}")
        except requests.exceptions.RequestException as e:
            log(f"{self.NODE_NAME}:Error sending notification:\n{e}")

