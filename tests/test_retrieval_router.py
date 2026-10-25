import unittest

import faiss
import numpy as np

from src.embeddings.retrieval_router import SchemeRetrievalRouter


class FakeModel:
    def encode(self, _texts, **_kwargs):
        return np.array([[1.0, 0.0]], dtype=np.float32)


def flat_index(vectors):
    index = faiss.IndexFlatIP(2)
    index.add(np.array(vectors, dtype=np.float32))
    return index


class RetrievalRouterTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "chunk_id": "aabcs_intro",
                "text": "intro",
                "metadata": {"source_file": "aabcs.txt", "scheme_name": "Annal Ambedkar Business Champions Scheme"},
            },
            {
                "chunk_id": "aabcs_eligibility",
                "text": "eligibility",
                "metadata": {"source_file": "aabcs.txt", "scheme_name": "Annal Ambedkar Business Champions Scheme"},
            },
            {
                "chunk_id": "farmer_benefit",
                "text": "farmer benefit",
                "metadata": {"source_file": "farmer.txt", "scheme_name": "Farmer Support Scheme"},
            },
        ]
        self.cards = [
            {"source_file": "aabcs.txt", "scheme_name": "Annal Ambedkar Business Champions Scheme", "abbreviation": "AABCS"},
            {"source_file": "farmer.txt", "scheme_name": "Farmer Support Scheme", "abbreviation": None},
        ]
        self.router = SchemeRetrievalRouter(
            chunks=self.chunks,
            chunk_index=flat_index([[0.6, 0.8], [0.95, 0.05], [0.8, 0.2]]),
            cards=self.cards,
            scheme_index=flat_index([[0.7, 0.3], [0.99, 0.01]]),
            model=FakeModel(),
        )

    def test_exact_match_limits_chunks_to_named_scheme(self):
        result = self.router.retrieve("What is the eligibility for AABCS?", chunks_per_scheme=2)
        self.assertEqual(result.route, "exact_scheme")
        self.assertEqual(result.schemes[0].source_file, "aabcs.txt")
        self.assertEqual(
            [chunk.chunk_id for chunk in result.chunks_by_source["aabcs.txt"]],
            ["aabcs_eligibility", "aabcs_intro"],
        )

    def test_broad_question_uses_scheme_card_ranking(self):
        result = self.router.retrieve("Which schemes help farmers?", scheme_limit=1)
        self.assertEqual(result.route, "broad_scheme_discovery")
        self.assertEqual(result.schemes[0].source_file, "farmer.txt")
        self.assertEqual(
            result.chunks_by_source["farmer.txt"][0].chunk_id,
            "farmer_benefit",
        )


if __name__ == "__main__":
    unittest.main()
