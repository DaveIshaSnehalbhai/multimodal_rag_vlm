"""Google Translate wrapper for Indic language output."""
from typing import Dict


class Translator:
    LANGS = ["hi", "bn", "mr", "gu"]

    def translate(self, text: str, lang: str) -> str:
        from deep_translator import GoogleTranslator
        try:
            return GoogleTranslator(source="en", target=lang).translate(text) or ""
        except Exception as e:
            print(f"[Translator] {lang}: {e}"); return ""

    def translate_all(self, caption_en: str) -> Dict[str, str]:
        return {lang: self.translate(caption_en, lang) for lang in self.LANGS}
