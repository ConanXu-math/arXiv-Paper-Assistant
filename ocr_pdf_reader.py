"""OCR-based PDF Reader for math papers.

Uses PyMuPDF to render each page as an image, then calls the edusys OCR API
to extract text with LaTeX formulas preserved — much better than pypdf's
extract_text() for math-heavy documents.
"""

import base64
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Union
from uuid import uuid4

import fitz  # PyMuPDF
import httpx
from agno.knowledge.chunking.strategy import ChunkingStrategy
from agno.knowledge.document.base import Document
from agno.knowledge.reader.pdf_reader import BasePDFReader
from agno.knowledge.types import ContentType

logger = logging.getLogger(__name__)

DEFAULT_OCR_URL = "https://edusys5.sii.edu.cn/ocr"


def _ocr_page_image(b64_png: str, ocr_url: str, timeout: float) -> Optional[str]:
    """Send a base64-encoded PNG to the OCR API and return recognised text."""
    try:
        resp = httpx.post(
            ocr_url,
            json={"image_base64": b64_png},
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                return data.get("result", "")
        logger.warning("OCR request failed (status %s): %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.warning("OCR request error: %s", e)
    return None


class OcrPDFReader(BasePDFReader):
    """PDF reader that uses OCR to extract text with LaTeX math formulas.

    After calling read(), the per-page OCR text is available via
    ``last_ocr_pages`` for downstream structure extraction.
    """

    def __init__(
        self,
        ocr_url: str = DEFAULT_OCR_URL,
        dpi: int = 200,
        max_workers: int = 4,
        request_timeout: float = 60.0,
        *,
        split_on_pages: bool = True,
        chunking_strategy: Optional[ChunkingStrategy] = None,
        **kwargs,
    ):
        self.ocr_url = ocr_url
        self.dpi = dpi
        self.max_workers = max_workers
        self.request_timeout = request_timeout
        self.last_ocr_pages: list[str] = []
        super().__init__(
            split_on_pages=split_on_pages,
            chunking_strategy=chunking_strategy,
            **kwargs,
        )

    @classmethod
    def get_supported_content_types(cls) -> List[ContentType]:
        return [ContentType.PDF]

    def _render_page_to_b64(self, page: fitz.Page) -> str:
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return base64.b64encode(pix.tobytes("png")).decode("utf-8")

    def _ocr_single_page(self, page_index: int, b64_png: str) -> tuple[int, str]:
        """OCR one page; returns (page_index, text)."""
        text = _ocr_page_image(b64_png, self.ocr_url, self.request_timeout)
        if text is None:
            logger.warning("Page %d: OCR failed, returning empty", page_index + 1)
            return page_index, ""
        return page_index, text

    def _create_documents_with_metadata(
        self,
        pdf_content: List[str],
        doc_name: str,
        page_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[Document]:
        """Create Documents with per-page structural metadata, then chunk."""
        documents: List[Document] = []
        for i, page_text in enumerate(pdf_content):
            page_num = i + 1
            meta: Dict[str, Any] = {"page": page_num}
            if page_metadata and i in page_metadata:
                meta.update(page_metadata[i])
            documents.append(
                Document(
                    name=doc_name,
                    id=str(uuid4()),
                    meta_data=meta,
                    content=page_text,
                )
            )
        if self.chunk:
            return self._build_chunked_documents(documents)
        return documents

    def read(
        self,
        pdf: Optional[Union[str, Path, IO[Any]]] = None,
        name: Optional[str] = None,
        password: Optional[str] = None,
        page_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[Document]:
        """Read a PDF via OCR.

        Args:
            page_metadata: optional dict mapping page index (0-based) to extra
                metadata fields (e.g. section, element_types) that will be
                attached to each page Document before chunking.
        """
        if pdf is None:
            logger.error("No pdf provided")
            return []

        doc_name = self._get_doc_name(pdf, name)
        logger.debug("Reading (OCR): %s", doc_name)

        doc = fitz.open(pdf)
        total_pages = len(doc)
        logger.info("OCR: %s — %d pages at %d dpi, max_workers=%d", doc_name, total_pages, self.dpi, self.max_workers)

        b64_images: list[tuple[int, str]] = []
        for i, page in enumerate(doc):
            b64_images.append((i, self._render_page_to_b64(page)))
        doc.close()

        pdf_content: list[Optional[str]] = [None] * total_pages

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._ocr_single_page, idx, b64): idx
                for idx, b64 in b64_images
            }
            for future in as_completed(futures):
                idx, text = future.result()
                pdf_content[idx] = text
                logger.info("OCR: page %d/%d done", idx + 1, total_pages)

        pdf_content = [t if t is not None else "" for t in pdf_content]

        # Expose raw OCR pages for downstream structure extraction
        self.last_ocr_pages = list(pdf_content)

        return self._create_documents_with_metadata(
            pdf_content,
            doc_name,
            page_metadata=page_metadata,
        )
