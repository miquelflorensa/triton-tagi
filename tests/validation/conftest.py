"""Shared fixtures and guards for validation tests against cuTAGI (pytagi)."""

import pytest

pytagi = pytest.importorskip("pytagi", reason="cuTAGI (pytagi) not installed")
