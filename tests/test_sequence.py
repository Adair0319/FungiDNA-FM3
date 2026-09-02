import unittest
import tempfile
from pathlib import Path

from fungidna.invoke.sequence import read_sequences, first_record_valid, SUPPORTED_EXTENSIONS


def _write(content, suffix=".fasta"):
    f = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


class TestReadSequences(unittest.TestCase):
    def test_fasta_two_records(self):
        p = _write(">seq1\nACGT\n>seq2 description\nTTTT\n")
        recs = read_sequences(p)
        self.assertEqual([r[0] for r in recs], [">seq1", ">seq2 description"])
        self.assertEqual([r[1] for r in recs], ["ACGT", "TTTT"])

    def test_fastq(self):
        p = _write("@r1\nACGT\n+\nIIII\n", suffix=".fastq")
        recs = read_sequences(p)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0][1], "ACGT")

    def test_invalid_first_line(self):
        p = _write("not a sequence file\n")
        with self.assertRaises(ValueError):
            read_sequences(p)

    def test_first_record_valid(self):
        self.assertTrue(first_record_valid(_write(">x\nACGT\n")))
        self.assertTrue(first_record_valid(_write("@x\nACGT\n+\nIIII\n", suffix=".fastq")))
        self.assertFalse(first_record_valid(_write("junk\n")))

    def test_supported_extensions(self):
        self.assertEqual(SUPPORTED_EXTENSIONS, {".fa", ".fasta", ".fq", ".fastq"})
