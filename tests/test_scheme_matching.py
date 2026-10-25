import unittest

from src.embeddings.scheme_matching import match_scheme, normalize


CARDS = [
    {
        "source_file": "aabcs.txt",
        "scheme_name": "Annal Ambedkar Business Champions Scheme",
        "abbreviation": "AABCS",
    },
    {
        "source_file": "aaas-assam.txt",
        "scheme_name": "Atal Amrit Abhiyan Scheme",
        "abbreviation": None,
    },
    {
        "source_file": "agr3fmsstf.txt",
        "scheme_name": "AGR 3 (Farm Mechanization) Scheme For ST Farmers",
        "abbreviation": None,
    },
]


class SchemeMatchingTests(unittest.TestCase):
    def test_normalize_removes_punctuation_and_case(self):
        self.assertEqual(normalize("AGR-3 (Farm)"), "agr 3 farm")

    def test_exact_title_match(self):
        result = match_scheme("What is the eligibility for Atal Amrit Abhiyan Scheme?", CARDS)
        self.assertEqual(result.source_file, "aaas-assam.txt")
        self.assertEqual(result.match_type, "exact_title")

    def test_exact_abbreviation_match(self):
        result = match_scheme("How can I apply under AABCS?", CARDS)
        self.assertEqual(result.source_file, "aabcs.txt")
        self.assertEqual(result.match_type, "exact_abbreviation")

    def test_fuzzy_title_match_for_typo(self):
        result = match_scheme("Who is eligible for Annal Ambedkar Busines Champions Scheme?", CARDS)
        self.assertEqual(result.source_file, "aabcs.txt")
        self.assertEqual(result.match_type, "fuzzy_title")

    def test_unrelated_question_has_no_exact_scheme_match(self):
        self.assertIsNone(match_scheme("Which schemes help low-income students?", CARDS))


if __name__ == "__main__":
    unittest.main()
