import csv
import tempfile
import unittest
from pathlib import Path

from evo_rag_hpo.evaluate import LOG_FIELDNAMES


class LoggingContractTests(unittest.TestCase):
    def test_computation_log_contract_contains_required_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "computation_log.csv"
            row = {field: "" for field in LOG_FIELDNAMES}
            row.update(
                {
                    "Hash Id": "20195b88",
                    "chunk_size": 256,
                    "top_k": 4,
                    "model_name": "granite3.3:2b",
                    "f1_score": 0.25,
                }
            )

            with output.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=LOG_FIELDNAMES)
                writer.writeheader()
                writer.writerow(row)

            with output.open("r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                self.assertEqual(reader.fieldnames, LOG_FIELDNAMES)
                loaded = next(reader)

            self.assertEqual(loaded["Hash Id"], "20195b88")
            self.assertEqual(loaded["model_name"], "granite3.3:2b")


if __name__ == "__main__":
    unittest.main()

