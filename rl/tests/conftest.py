"""Shared pytest fixtures.

None of these tests depend on a real server/bridge process or a real `/nn` checkpoint -- every
fixture/factory (see `factories.py`) builds small, in-memory-shaped objects from scratch, and the
Mineflayer bridge process is mocked (see `test_match_coordinator.py::FakeBridgeProcess`), so the
test suite is fully self-contained and fast.
"""

from __future__ import annotations
