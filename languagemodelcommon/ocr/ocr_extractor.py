class OCRExtractor:
    async def extract_text_with_textract_async(self, pdf_bytes: bytes) -> str:
        raise NotImplementedError("Method should be implemented by subclass")
