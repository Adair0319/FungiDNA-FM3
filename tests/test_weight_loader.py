import os
import unittest
from pathlib import Path

from fungidna.invoke.handlers.base import WeightLoader, WeightsNotFoundError


class TestWeightLoader(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("FUNGIDNA_TAXONOMY_WEIGHTS", None)

    def test_registry_wins(self):
        wl = WeightLoader({"taxonomy": "/tmp/tax.pt"})
        self.assertEqual(wl.resolve("taxonomy"), Path("/tmp/tax.pt"))

    def test_env_fallback(self):
        os.environ["FUNGIDNA_TAXONOMY_WEIGHTS"] = "/tmp/env.pt"
        wl = WeightLoader({})
        self.assertEqual(wl.resolve("taxonomy"), Path("/tmp/env.pt"))

    def test_missing_raises(self):
        wl = WeightLoader({})
        with self.assertRaises(WeightsNotFoundError) as cm:
            wl.resolve("splice_site")
        self.assertEqual(cm.exception.task, "splice_site")
