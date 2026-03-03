"""OCR service refactored from ocr.py — uses Mistral OCR to extract text from PDFs."""

import base64
import json
import logging
from pathlib import Path

from mistralai import Mistral

from backend.config import settings
from backend.services.storage_service import (
    images_dir,
    images_list_path,
    preprint_md_path,
)

logger = logging.getLogger(__name__)


class OCRService:
    def __init__(self, api_key: str | None = None):
        self.client = Mistral(api_key=api_key or settings.mistral_api_key)

    @staticmethod
    def _encode_pdf(pdf_path: str | Path) -> str:
        with open(pdf_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def process_pdf(self, pdf_path: str | Path, key: str) -> str:
        """Run Mistral OCR on a PDF and save the results.

        Returns the extracted markdown text.
        """
        logger.info("Starting OCR for key=%s, file=%s", key, pdf_path)
        base64_pdf = self._encode_pdf(pdf_path)

        ocr_response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}",
            },
            table_format="html",
            include_image_base64=True,
        )

        # Combine all page texts
        full_text_parts: list[str] = []
        all_images: list[dict] = []

        for page in ocr_response.pages:
            full_text_parts.append(page.markdown)

            # Save embedded images
            if page.images:
                for img in page.images:
                    img_stem = Path(img.id).stem

                    # Decode and save image if base64 present
                    img_filename = f"{img_stem}.png"  # default
                    if img.image_base64:
                        img_b64 = img.image_base64
                        # Strip data URI prefix if present (e.g. "data:image/jpeg;base64,...")
                        if "," in img_b64 and img_b64.startswith("data:"):
                            img_b64 = img_b64.split(",", 1)[1]
                        img_bytes = base64.b64decode(img_b64)
                        # Detect actual format from magic bytes
                        ext = ".png" if img_bytes[:4] == b"\x89PNG" else ".jpg"
                        img_filename = f"{img_stem}{ext}"
                        img_path = images_dir(key) / img_filename
                        img_path.write_bytes(img_bytes)

                    img_data = {
                        "id": img_filename,
                        "top_left_x": img.top_left_x,
                        "top_left_y": img.top_left_y,
                        "bottom_right_x": img.bottom_right_x,
                        "bottom_right_y": img.bottom_right_y,
                    }
                    all_images.append(img_data)

        full_text = "\n\n".join(full_text_parts)

        # Save markdown
        preprint_md_path(key).write_text(full_text, encoding="utf-8")

        # Save images list
        if all_images:
            images_list_path(key).write_text(
                json.dumps(all_images, indent=2), encoding="utf-8"
            )

        logger.info("OCR complete for key=%s, pages=%d", key, len(ocr_response.pages))
        return full_text
