from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def snapshot_path() -> Path:
    return FIXTURES / "confusing_tools.json"


@pytest.fixture
def tool_set(snapshot_path: Path):
    from mispick.sources.snapshot import load_snapshot

    return load_snapshot(snapshot_path)


@pytest.fixture
def fixture_server():
    from tests.fixtures.fixture_server import mcp

    return mcp
