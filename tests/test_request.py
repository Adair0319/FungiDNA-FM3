import enum
import unittest
from pathlib import Path

from fungidna.invoke.request import Task, StructuredRequest, task_from_string


class TestTaskEnum(unittest.TestCase):
    def test_values_match_canonical(self):
        self.assertEqual(Task.REPRESENTATION.value, "representation")
        self.assertEqual(Task.TAXONOMY.value, "taxonomy")
        self.assertEqual(Task.CODING_POTENTIAL.value, "coding_potential")
        self.assertEqual(Task.SPLICE_SITE.value, "splice_site")
        self.assertEqual(Task.BGC_BOUNDARY.value, "bgc_boundary")

    def test_task_from_string(self):
        self.assertIs(task_from_string("taxonomy"), Task.TAXONOMY)

    def test_task_from_string_rejects_unknown(self):
        with self.assertRaises(ValueError):
            task_from_string("promoter")


class TestStructuredRequest(unittest.TestCase):
    def test_construct_and_fields(self):
        r = StructuredRequest(task=Task.TAXONOMY, file_path=Path("/tmp/x.fasta"))
        self.assertEqual(r.task, Task.TAXONOMY)
        self.assertEqual(r.file_path, Path("/tmp/x.fasta"))

    def test_frozen(self):
        r = StructuredRequest(task=Task.TAXONOMY, file_path=Path("/tmp/x.fasta"))
        with self.assertRaises(AttributeError):
            r.task = Task.SPLICE_SITE
