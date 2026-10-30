import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.maas_rate_limit import wait_for_slot


class MaasRateLimitTest(unittest.TestCase):
    def test_sequential_calls_reserve_distinct_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rate"
            with patch.dict(os.environ, {
                    "MAAS_RATE_LIMIT_FILE": str(path),
                    "MAAS_REQUESTS_PER_MINUTE": "6000"}):
                wait_for_slot()
                first = float(path.read_text())
                wait_for_slot()
                second = float(path.read_text())
            self.assertGreater(second, first)
            self.assertAlmostEqual(second - first, 0.01, delta=0.005)


if __name__ == "__main__":
    unittest.main()
