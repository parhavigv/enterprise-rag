"""Excel (.xlsx / .xlsm) parser.

Reads each worksheet and flattens its rows into readable records so tabular
data (spreadsheets, reports, budgets) is indexed queryably. Handles merged
cells and multi-row headers gracefully; each sheet becomes one
:class:`Document` with metadata identifying the file, sheet, and row count.

Requires ``openpyxl`` (a hard dependency added to the project).
"""

from __future__ import annotations

from pathlib import Path

from llama_index.core import Document

SUPPORTED_EXTENSIONS = {".xlsx", ".xlsm"}


def parse_xlsx(path: str, *, max_sheets: int | None = None) -> list[Document]:
    """Parse an Excel workbook into one document per worksheet."""
    from openpyxl import load_workbook

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{p.suffix}'. Expected one of {SUPPORTED_EXTENSIONS}."
        )

    try:
        wb = load_workbook(p, data_only=True, read_only=False)
    except Exception as e:  # noqa: BLE001 - surface corrupt/incompatible workbooks clearly
        raise ValueError(f"Failed to open Excel workbook '{path}': {e}") from e

    docs: list[Document] = []
    sheet_names = wb.sheetnames
    if max_sheets:
        sheet_names = sheet_names[:max_sheets]

    for sheet_name in sheet_names:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        while rows and all(cell is None or str(cell).strip() == "" for cell in rows[-1]):
            rows.pop()  # drop fully-empty trailing rows
        if not rows:
            continue

        records = [_row_to_text(row) for row in rows if any(c is not None for c in row)]
        text = "\n".join(records).strip()
        if not text:
            continue

        docs.append(
            Document(
                text=text,
                metadata={
                    "format": "xlsx",
                    "source": str(path),
                    "file_path": str(path),
                    "sheet": sheet_name,
                    "rows": len(records),
                },
            )
        )

    wb.close()
    if not docs:
        raise ValueError(f"Excel workbook '{path}' contained no readable data.")
    return docs


def _row_to_text(row: tuple) -> str:
    """Render a spreadsheet row as a colon-separated record of its cells."""
    parts = []
    for cell in row:
        if cell is None:
            continue
        value = str(cell).strip()
        if value:
            parts.append(value)
    return " ; ".join(parts)
