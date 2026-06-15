import unittest

from evo_rag_hpo.index import collection_has_documents


class FakeVectorStore:
    def __init__(self, ids=None, should_raise=False):
        self.ids = ids or []
        self.should_raise = should_raise

    def get(self, limit=1):
        if self.should_raise:
            raise RuntimeError("collection unavailable")
        return {"ids": self.ids[:limit]}


class IndexTests(unittest.TestCase):
    def test_collection_has_documents_detects_existing_ids(self):
        self.assertTrue(collection_has_documents(FakeVectorStore(ids=["doc-1"])))

    def test_collection_has_documents_handles_empty_collection(self):
        self.assertFalse(collection_has_documents(FakeVectorStore()))

    def test_collection_has_documents_handles_chroma_errors(self):
        self.assertFalse(collection_has_documents(FakeVectorStore(should_raise=True)))


if __name__ == "__main__":
    unittest.main()
