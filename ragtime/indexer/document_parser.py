"""
Document Content Parser

Extracts text content from various document formats:
- PDF (.pdf)
- Word (.docx, .doc)
- Excel (.xlsx, .xls)
- PowerPoint (.pptx)
- OpenDocument (.odt, .ods, .odp)
- Plain text (.txt, .md, .rst, .json, .xml, .html, .csv)
- Code files (.py, .js, .ts, etc.)
"""

import io
from pathlib import Path
from typing import Optional

from ragtime.core.logging import get_logger

logger = get_logger(__name__)


def extract_text_from_file(file_path: Path, content: Optional[bytes] = None) -> str:
    """
    Extract text content from a file based on its extension.

    Args:
        file_path: Path to the file
        content: Optional pre-loaded file content (bytes)

    Returns:
        Extracted text content as string
    """
    suffix = file_path.suffix.lower()

    # Load content if not provided
    if content is None:
        try:
            content = file_path.read_bytes()
        except Exception as e:
            logger.warning(f"Failed to read file {file_path}: {e}")
            return ""

    # Route to appropriate parser
    try:
        if suffix == ".pdf":
            return _extract_pdf(content)
        elif suffix == ".docx":
            return _extract_docx(content)
        elif suffix == ".doc":
            return _extract_doc_legacy(file_path, content)
        elif suffix == ".xlsx":
            return _extract_xlsx(content)
        elif suffix == ".xls":
            return _extract_xls(content)
        elif suffix == ".pptx":
            return _extract_pptx(content)
        elif suffix == ".odt":
            return _extract_odt(content)
        elif suffix == ".ods":
            return _extract_ods(content)
        elif suffix == ".odp":
            return _extract_odp(content)
        else:
            # Plain text files
            return _extract_text(content)
    except Exception as e:
        logger.warning(f"Failed to extract text from {file_path}: {e}")
        return ""


def _extract_pdf(content: bytes) -> str:
    """Extract text from PDF file."""
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf not installed, cannot extract PDF content")
        return ""

    try:
        reader = PdfReader(io.BytesIO(content))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.warning(f"PDF extraction error: {e}")
        return ""


def _extract_docx(content: bytes) -> str:
    """Extract text from Word DOCX file."""
    try:
        from docx import Document
    except ImportError:
        logger.warning("python-docx not installed, cannot extract DOCX content")
        return ""

    try:
        doc = Document(io.BytesIO(content))
        text_parts = []

        # Extract paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)

        # Extract tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    text_parts.append(row_text)

        return "\n\n".join(text_parts)
    except Exception as e:
        logger.warning(f"DOCX extraction error: {e}")
        return ""


def _extract_doc_legacy(file_path: Path, content: bytes) -> str:
    """Extract text from legacy Word DOC file."""
    # Legacy .doc files are more complex
    # Try antiword if available, otherwise skip
    import subprocess

    try:
        result = subprocess.run(
            ["antiword", str(file_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout
    except FileNotFoundError:
        logger.debug("antiword not installed, cannot extract legacy DOC content")
    except Exception as e:
        logger.warning(f"DOC extraction error: {e}")

    return ""


def _extract_xlsx(content: bytes) -> str:
    """Extract text from Excel XLSX file."""
    try:
        from openpyxl import load_workbook
    except ImportError:
        logger.warning("openpyxl not installed, cannot extract XLSX content")
        return ""

    try:
        wb = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        text_parts = []

        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            text_parts.append(f"## Sheet: {sheet_name}")

            for row in sheet.iter_rows():
                row_values = []
                for cell in row:
                    if cell.value is not None:
                        row_values.append(str(cell.value))
                if row_values:
                    text_parts.append(" | ".join(row_values))

        wb.close()
        return "\n".join(text_parts)
    except Exception as e:
        logger.warning(f"XLSX extraction error: {e}")
        return ""


def _extract_xls(content: bytes) -> str:
    """Extract text from legacy Excel XLS file."""
    try:
        import xlrd
    except ImportError:
        logger.warning("xlrd not installed, cannot extract XLS content")
        return ""

    try:
        wb = xlrd.open_workbook(file_contents=content)
        text_parts = []

        for sheet in wb.sheets():
            text_parts.append(f"## Sheet: {sheet.name}")

            for row_idx in range(sheet.nrows):
                row_values = []
                for col_idx in range(sheet.ncols):
                    cell = sheet.cell(row_idx, col_idx)
                    if cell.value:
                        row_values.append(str(cell.value))
                if row_values:
                    text_parts.append(" | ".join(row_values))

        return "\n".join(text_parts)
    except Exception as e:
        logger.warning(f"XLS extraction error: {e}")
        return ""


def _extract_pptx(content: bytes) -> str:
    """Extract text from PowerPoint PPTX file."""
    try:
        from pptx import Presentation
    except ImportError:
        logger.warning("python-pptx not installed, cannot extract PPTX content")
        return ""

    try:
        prs = Presentation(io.BytesIO(content))
        text_parts = []

        for slide_num, slide in enumerate(prs.slides, 1):
            slide_texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_texts.append(shape.text)

            if slide_texts:
                text_parts.append(f"## Slide {slide_num}")
                text_parts.extend(slide_texts)

        return "\n\n".join(text_parts)
    except Exception as e:
        logger.warning(f"PPTX extraction error: {e}")
        return ""


def _extract_odt(content: bytes) -> str:
    """Extract text from OpenDocument Text file."""
    try:
        from odf import text as odf_text
        from odf.opendocument import load
    except ImportError:
        logger.warning("odfpy not installed, cannot extract ODT content")
        return ""

    try:
        doc = load(io.BytesIO(content))
        text_parts = []

        for para in doc.getElementsByType(odf_text.P):
            text = ""
            for node in para.childNodes:
                if hasattr(node, "data"):
                    text += node.data
            if text.strip():
                text_parts.append(text)

        return "\n\n".join(text_parts)
    except Exception as e:
        logger.warning(f"ODT extraction error: {e}")
        return ""


def _extract_ods(content: bytes) -> str:
    """Extract text from OpenDocument Spreadsheet file."""
    try:
        from odf import table as odf_table
        from odf import text as odf_text
        from odf.opendocument import load
    except ImportError:
        logger.warning("odfpy not installed, cannot extract ODS content")
        return ""

    try:
        doc = load(io.BytesIO(content))
        text_parts = []

        for sheet in doc.getElementsByType(odf_table.Table):
            sheet_name = sheet.getAttribute("name") or "Sheet"
            text_parts.append(f"## Sheet: {sheet_name}")

            for row in sheet.getElementsByType(odf_table.TableRow):
                row_values = []
                for cell in row.getElementsByType(odf_table.TableCell):
                    cell_text = ""
                    for p in cell.getElementsByType(odf_text.P):
                        for node in p.childNodes:
                            if hasattr(node, "data"):
                                cell_text += node.data
                    if cell_text:
                        row_values.append(cell_text)
                if row_values:
                    text_parts.append(" | ".join(row_values))

        return "\n".join(text_parts)
    except Exception as e:
        logger.warning(f"ODS extraction error: {e}")
        return ""


def _extract_odp(content: bytes) -> str:
    """Extract text from OpenDocument Presentation file."""
    try:
        from odf import draw as odf_draw
        from odf import text as odf_text
        from odf.opendocument import load
    except ImportError:
        logger.warning("odfpy not installed, cannot extract ODP content")
        return ""

    try:
        doc = load(io.BytesIO(content))
        text_parts = []

        for page_num, page in enumerate(doc.getElementsByType(odf_draw.Page), 1):
            page_texts = []
            for frame in page.getElementsByType(odf_draw.Frame):
                for text_box in frame.getElementsByType(odf_draw.TextBox):
                    for p in text_box.getElementsByType(odf_text.P):
                        text = ""
                        for node in p.childNodes:
                            if hasattr(node, "data"):
                                text += node.data
                        if text.strip():
                            page_texts.append(text)

            if page_texts:
                text_parts.append(f"## Slide {page_num}")
                text_parts.extend(page_texts)

        return "\n\n".join(text_parts)
    except Exception as e:
        logger.warning(f"ODP extraction error: {e}")
        return ""


def _extract_text(content: bytes) -> str:
    """Extract text from plain text file."""
    # Try common encodings
    for encoding in ["utf-8", "latin-1", "cp1252", "ascii"]:
        try:
            return content.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue

    # Last resort: decode with errors replaced
    return content.decode("utf-8", errors="replace")


# Supported extensions for document parsing
DOCUMENT_EXTENSIONS = {
    # Office documents
    ".pdf",
    ".docx", ".doc",
    ".xlsx", ".xls",
    ".pptx",
    # OpenDocument
    ".odt", ".ods", ".odp",
    # Plain text
    ".txt", ".md", ".rst",
    ".json", ".xml", ".html", ".htm",
    ".csv", ".tsv",
    # Code
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".c", ".cpp", ".h", ".hpp",
    ".go", ".rs", ".rb", ".php",
    ".sql", ".sh", ".bash", ".zsh",
    ".yaml", ".yml", ".toml", ".ini", ".cfg",
}


def is_supported_document(file_path: Path) -> bool:
    """Check if a file type is supported for text extraction."""
    return file_path.suffix.lower() in DOCUMENT_EXTENSIONS
