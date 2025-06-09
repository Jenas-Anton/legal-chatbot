# app/text_processor.py
import textwrap

class TextProcessor:
    def chunk_text(self, text: str, chunk_size: int = 2000) -> list:
        return textwrap.wrap(text, chunk_size)