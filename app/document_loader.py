# app/document_loader.py
from pathlib import Path
from PyPDF2 import PdfReader
import docx

class DocumentLoader:
    def load_document(self, file_path: str) -> str:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(file_path)
        elif suffix == ".docx":
            return self._load_docx(file_path)
        elif suffix == ".txt":
            return self._load_txt(file_path)
        else:
            raise ValueError("Unsupported file format")

    def _load_pdf(self, file_path: str) -> str:
        text = ""
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    def _load_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    def _load_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()