"""Integration tests for the axon_object_flow MCP tool.

Uses a real KuzuBackend populated by run_pipeline so that execute_raw
queries are exercised end-to-end, matching production behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from axon.core.ingestion.pipeline import run_pipeline
from axon.core.storage.kuzu_backend import KuzuBackend
from axon.mcp.tools import handle_object_flow


# ---------------------------------------------------------------------------
# Shared fixture: a minimal two-file JS project where a.js instantiates
# UserService defined in b.js, then indexed into a real KuzuBackend.
# ---------------------------------------------------------------------------

@pytest.fixture()
def storage(tmp_path: Path) -> KuzuBackend:
    """Initialised KuzuBackend written to a temp directory."""
    db_path = tmp_path / "axon_db"
    backend = KuzuBackend()
    backend.initialize(db_path)
    yield backend
    backend.close()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """Minimal JS repo: a.js instantiates UserService from b.js."""
    a_js = tmp_path / "a.js"
    b_js = tmp_path / "b.js"
    a_js.write_text(
        "import { UserService } from './b';\n"
        "const u = new UserService();\n",
        encoding="utf-8",
    )
    b_js.write_text("export class UserService {}\n", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def indexed_storage(repo: Path, storage: KuzuBackend) -> KuzuBackend:
    """Run the full pipeline and persist into storage."""
    run_pipeline(repo, storage, embeddings=False)
    return storage


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetObjectFlow:
    def test_known_class_returns_one_result(
        self, indexed_storage: KuzuBackend
    ) -> None:
        """handle_object_flow returns exactly one entry for UserService."""
        result = handle_object_flow(indexed_storage, "UserService")

        # Must not be an error message.
        assert not result.startswith("Error:"), result
        assert not result.startswith("Class 'UserService' not found"), result

        # Exactly one instantiation site.
        lines = [ln for ln in result.splitlines() if ln.strip().startswith("1.")]
        assert len(lines) == 1, f"Expected 1 site line, got:\n{result}"

        # The site must reference a.js.
        assert "a.js" in lines[0], f"Expected a.js in site line: {lines[0]}"

    def test_known_class_reports_correct_file_in_site(
        self, indexed_storage: KuzuBackend
    ) -> None:
        """The site line in the output references a.js."""
        result = handle_object_flow(indexed_storage, "UserService")
        site_line = next(
            ln for ln in result.splitlines() if ln.strip().startswith("1.")
        )
        # The instantiation is in a.js (module-level, source is the FILE node).
        assert "a.js" in site_line, (
            f"Expected 'a.js' in site line, got: {site_line}"
        )

    def test_known_class_reports_defined_in(
        self, indexed_storage: KuzuBackend
    ) -> None:
        """The 'Defined in' header must reference b.js."""
        result = handle_object_flow(indexed_storage, "UserService")
        assert "b.js" in result, (
            f"Expected 'b.js' in result for class_defined_in, got:\n{result}"
        )

    def test_nonexistent_class_returns_not_found(
        self, indexed_storage: KuzuBackend
    ) -> None:
        """handle_object_flow returns a safe not-found message for unknown classes."""
        result = handle_object_flow(indexed_storage, "NonExistentClass")
        assert "not found" in result.lower(), (
            f"Expected a 'not found' message, got:\n{result}"
        )
        # Must not raise and must not be empty.
        assert result.strip() != ""

    def test_empty_class_name_returns_error(
        self, indexed_storage: KuzuBackend
    ) -> None:
        """Passing an empty string must return an error, not crash."""
        result = handle_object_flow(indexed_storage, "")
        assert result.startswith("Error:"), f"Expected Error: prefix, got: {result}"
