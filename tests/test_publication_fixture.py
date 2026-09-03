import unittest

from papers import PAPERS


class PublicationFixtureTests(unittest.TestCase):
    def test_inputs_are_explicitly_synthetic_and_not_cited_as_real_papers(self):
        self.assertEqual(len(PAPERS), 5)
        for paper in PAPERS:
            self.assertEqual(paper["data_type"], "synthetic_test_fixture")
            self.assertEqual(paper["doi"], "")
            self.assertEqual(paper["authors"], [])
            self.assertIn("SYNTHETIC TEST ONLY", paper["title"])
            self.assertIn("not scientific evidence", paper["abstract"])


if __name__ == "__main__":
    unittest.main()
