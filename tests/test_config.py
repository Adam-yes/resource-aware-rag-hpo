import unittest

from evo_rag_hpo.config import DEFAULT_CONFIG, decode_individual, genotype_hash, search_space_limits


class ConfigTests(unittest.TestCase):
    def test_decode_individual_maps_genotype_to_parameters(self):
        decoded = decode_individual([1, 0, 4, 3, 7], DEFAULT_CONFIG)

        self.assertEqual(
            decoded,
            {
                "chunk_size": 256,
                "chunk_overlap": 0,
                "top_k": 4,
                "temperature": 0.3,
                "model_name": "granite3.3:2b",
            },
        )

    def test_decode_individual_rejects_out_of_bounds_gene(self):
        limits = search_space_limits(DEFAULT_CONFIG)

        with self.assertRaises(IndexError):
            decode_individual([limits[0] + 1, 0, 0, 0, 0], DEFAULT_CONFIG)

    def test_genotype_hash_is_stable(self):
        self.assertEqual(genotype_hash([1, 0, 4, 3, 7]), genotype_hash([1, 0, 4, 3, 7]))
        self.assertNotEqual(genotype_hash([1, 0, 4, 3, 7]), genotype_hash([1, 0, 4, 3, 8]))


if __name__ == "__main__":
    unittest.main()

