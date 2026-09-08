"""Tests for the Excel (.xlsx) parser and ACL metadata tagging at ingestion."""

from __future__ import annotations

import pytest

from app.auth.models import ACLMetadata, ClearanceLevel
from ingestion.parsers import PARSERS


class TestXlsxParser:
    def test_xlsx_is_registered(self):
        assert "xlsx" in PARSERS
        assert PARSERS["xlsx"].__name__ == "parse_xlsx"

    def test_parse_xlsx_builds_sheet_documents(self, tmp_path):
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "Budget"
        ws.append(["Department", "Q4 Budget", "Headcount"])
        ws.append(["Engineering", 1200000, 80])
        ws.append(["Finance", 400000, 12])
        wb.save(tmp_path / "budget.xlsx")

        docs = PARSERS["xlsx"](str(tmp_path / "budget.xlsx"))
        assert len(docs) == 1
        assert docs[0].metadata["format"] == "xlsx"
        assert docs[0].metadata["sheet"] == "Budget"
        assert docs[0].metadata["rows"] == 3
        assert "Engineering" in docs[0].text
        assert "1200000" in docs[0].text

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PARSERS["xlsx"](str(tmp_path / "nope.xlsx"))


class TestACLMetadataDefaults:
    def test_merge_defaults_applies_department_and_clearance(self):
        meta = ACLMetadata.merge_defaults(
            {"source": "x.pdf"},
            {"department": "finance", "clearance_level": "confidential", "owner": "alice"},
        )
        assert meta["department"] == "FINANCE"
        assert meta["clearance_level"] == "CONFIDENTIAL"
        assert meta["owner"] == "alice"
        assert meta["source"] == "x.pdf"

    def test_merge_defaults_keeps_existing_values(self):
        meta = ACLMetadata.merge_defaults(
            {"department": "hr", "clearance_level": "internal"},
            {"department": "finance", "clearance_level": "secret"},
        )
        assert meta["department"] == "FINANCE"  # defaults win
        assert meta["clearance_level"] == "SECRET"  # defaults win

    def test_merge_defaults_empty_defaults_yields_public(self):
        meta = ACLMetadata.merge_defaults({"source": "x.pdf"}, {})
        assert meta["department"] == "PUBLIC"
        assert meta["clearance_level"] == ClearanceLevel.PUBLIC.name

    def test_from_metadata_coerces_values(self):
        acl = ACLMetadata.from_metadata(
            {"department": "legal", "clearance_level": "secret", "owner": "bob"}
        )
        assert acl.department == "LEGAL"
        assert acl.clearance_level == ClearanceLevel.SECRET
        assert acl.owner == "bob"

    def test_unknown_clearance_falls_back_to_public_in_merge(self):
        meta = ACLMetadata.merge_defaults({}, {"clearance_level": "bogus"})
        assert meta["clearance_level"] == ClearanceLevel.PUBLIC.name
