import unittest

from fungidna.invoke.parser import parse_prompt


class TestParsePrompt(unittest.TestCase):
    def test_regex_extracts_task_and_abspath(self):
        d = parse_prompt("Please predict the splice sites using /home/ssd/test.fasta")
        self.assertEqual(d["task"], "splice_site")
        self.assertEqual(d["file_path"], "/home/ssd/test.fasta")

    def test_regex_extracts_representation(self):
        d = parse_prompt("generate sequence representations for /data/x.fa")
        self.assertEqual(d["task"], "representation")
        self.assertEqual(d["file_path"], "/data/x.fa")

    def test_missing_file_path_triggers_llm(self):
        seen = {}

        def fake_llm(text):
            seen["called"] = True
            return {"task": "taxonomy", "file_path": "/tmp/x.fastq"}

        d = parse_prompt("classify this genome", llm_extractor=fake_llm)
        self.assertTrue(seen.get("called"))
        self.assertEqual(d["task"], "taxonomy")
        self.assertEqual(d["file_path"], "/tmp/x.fastq")

    def test_missing_everything_without_llm(self):
        d = parse_prompt("hello there")
        self.assertIsNone(d["task"])
        self.assertIsNone(d["file_path"])
