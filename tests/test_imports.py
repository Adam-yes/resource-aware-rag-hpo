import importlib
import unittest


class ImportTests(unittest.TestCase):
    def test_public_modules_import_without_runtime_services(self):
        modules = [
            "evo_rag_hpo",
            "evo_rag_hpo.config",
            "evo_rag_hpo.elitism",
            "evo_rag_hpo.evaluate",
            "evo_rag_hpo.index",
            "evo_rag_hpo.logger",
            "evo_rag_hpo.optimize",
            "evo_rag_hpo.question_generation",
            "evo_rag_hpo.rag_chain_pipeline",
        ]

        for module in modules:
            importlib.import_module(module)


if __name__ == "__main__":
    unittest.main()
