"""OCR service refactored from ocr.py — uses Mistral OCR to extract text from PDFs.

Large PDFs are split into page-chunks before OCR: the Azure mistral-document-ai
deployment on the LiteLLM proxy rejects documents over 30 pages, so a 93-page
manuscript is processed as 30 + 30 + 30 + 3 and the per-chunk markdown/images
are stitched back together. Image IDs (``img-0``, ``img-1`` …) restart in every
chunk, so chunked output namespaces them (``c1-img-0`` …) to avoid collisions.
"""

import base64
import io
import json
import logging
from pathlib import Path

from mistralai import Mistral
from pypdf import PdfReader, PdfWriter

from backend.config import settings
from backend.services.storage_service import (
    images_dir,
    images_list_path,
    preprint_md_path,
)

logger = logging.getLogger(__name__)


class OCRService:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        # Defaults route through the LiteLLM proxy (same credentials as the
        # review agent). BYOK callers pass the submitter's own Mistral key,
        # the public Mistral base URL, and the public OCR model.
        self.client = Mistral(
            api_key=api_key or settings.litellm_api_key,
            server_url=base_url or settings.litellm_base_url,
        )
        self.model = model or settings.ocr_model

    @staticmethod
    def _encode_pdf(pdf_path: str | Path) -> str:
        with open(pdf_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _split_pdf(pdf_path: str | Path, pages_per_chunk: int) -> list[bytes]:
        """Split a PDF into chunks of at most ``pages_per_chunk`` pages.

        Returns a list of PDF byte blobs. A PDF that already fits in one chunk
        is returned as a single blob (its original bytes, re-serialized).
        """
        reader = PdfReader(str(pdf_path))
        n = len(reader.pages)
        chunks: list[bytes] = []
        for start in range(0, n, pages_per_chunk):
            writer = PdfWriter()
            for i in range(start, min(start + pages_per_chunk, n)):
                writer.add_page(reader.pages[i])
            buf = io.BytesIO()
            writer.write(buf)
            chunks.append(buf.getvalue())
        return chunks

    def _ocr_chunk(self, pdf_b64: str):
        return self.client.ocr.process(
            model=self.model,
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{pdf_b64}",
            },
            table_format="html",
            include_image_base64=True,
        )

    def _collect(self, ocr_response, key: str, id_prefix: str = "") -> tuple[list[str], list[dict]]:
        """Pull markdown + images out of one OCR response.

        ``id_prefix`` namespaces image IDs for multi-chunk runs; it is "" for
        single-chunk runs, preserving the original (unprefixed) behavior.
        """
        text_parts: list[str] = []
        images: list[dict] = []

        for page in ocr_response.pages:
            page_md = page.markdown

            if page.images:
                for img in page.images:
                    # Namespace per chunk (c0-, c1-, …) so IDs from different
                    # chunks (Mistral restarts img-0 each response) don't collide.
                    img_stem = Path(f"{id_prefix}{img.id}").stem
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

                    # Point the markdown reference at the actual saved filename
                    # (correct extension + per-chunk namespace), so refs resolve.
                    page_md = page_md.replace(img.id, img_filename)

                    images.append({
                        "id": img_filename,
                        "top_left_x": img.top_left_x,
                        "top_left_y": img.top_left_y,
                        "bottom_right_x": img.bottom_right_x,
                        "bottom_right_y": img.bottom_right_y,
                    })

            text_parts.append(page_md)

        return text_parts, images

    def process_pdf(self, pdf_path: str | Path, key: str) -> str:
        """Run Mistral OCR on a PDF and save the results.

        Returns the extracted markdown text. PDFs longer than
        ``settings.ocr_max_pages_per_request`` are split into chunks and
        processed sequentially.
        """
        logger.info("Starting OCR for key=%s, file=%s", key, pdf_path)

        chunk_size = settings.ocr_max_pages_per_request
        chunks = self._split_pdf(pdf_path, chunk_size)

        full_text_parts: list[str] = []
        all_images: list[dict] = []
        total_pages = 0

        if len(chunks) == 1:
            # Single chunk: OCR the original file bytes directly (unchanged path).
            ocr_response = self._ocr_chunk(self._encode_pdf(pdf_path))
            parts, images = self._collect(ocr_response, key, id_prefix="")
            full_text_parts.extend(parts)
            all_images.extend(images)
            total_pages = len(ocr_response.pages)
        else:
            logger.info(
                "Key=%s: PDF split into %d chunks of up to %d pages",
                key, len(chunks), chunk_size,
            )
            for idx, chunk_bytes in enumerate(chunks):
                chunk_b64 = base64.b64encode(chunk_bytes).decode("utf-8")
                ocr_response = self._ocr_chunk(chunk_b64)
                parts, images = self._collect(ocr_response, key, id_prefix=f"c{idx}-")
                full_text_parts.extend(parts)
                all_images.extend(images)
                total_pages += len(ocr_response.pages)
                logger.info(
                    "Key=%s: OCR'd chunk %d/%d (%d pages)",
                    key, idx + 1, len(chunks), len(ocr_response.pages),
                )

        full_text = "\n\n".join(full_text_parts)

        # Save markdown
        preprint_md_path(key).write_text(full_text, encoding="utf-8")

        # Save images list
        if all_images:
            images_list_path(key).write_text(
                json.dumps(all_images, indent=2), encoding="utf-8"
            )

        logger.info("OCR complete for key=%s, pages=%d", key, total_pages)
        return full_text
