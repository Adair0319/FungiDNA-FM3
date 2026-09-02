import unittest
from fungidna.invoke.tasks import CANONICAL_TASKS, TASK_ALIASES, normalize_task


class TestNormalizeTask(unittest.TestCase):
    def test_canonical_unchanged(self):
        for t in CANONICAL_TASKS:
            self.assertEqual(normalize_task(t), t)

    def test_aliases_map_to_canonical(self):
        self.assertEqual(normalize_task("representations"), "representation")
        self.assertEqual(normalize_task("embedding"), "representation")
        self.assertEqual(normalize_task("taxonomic classification"), "taxonomy")
        self.assertEqual(normalize_task("classification"), "taxonomy")
        self.assertEqual(normalize_task("coding potential"), "coding_potential")
        self.assertEqual(normalize_task("coding-potential"), "coding_potential")
        self.assertEqual(normalize_task("splice site"), "splice_site")
        self.assertEqual(normalize_task("splice-site"), "splice_site")
        self.assertEqual(normalize_task("BGC boundary"), "bgc_boundary")
        self.assertEqual(normalize_task("bgc"), "bgc_boundary")

    def test_unknown_returns_none(self):
        self.assertIsNone(normalize_task("promoter"))
        self.assertIsNone(normalize_task(""))
        self.assertIsNone(normalize_task(None))

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(normalize_task("  Splice Site "), "splice_site")
        self.assertEqual(normalize_task("CODING_POTENTIAL"), "coding_potential")
