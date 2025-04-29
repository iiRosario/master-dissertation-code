import unittest
import numpy as np
from env import * 
from entities.Annotator import Annotator


class TestAnnotator(unittest.TestCase):

    def setUp(self):
        self.annotator = Annotator(id=1, seed=42, num_classes=3)
        self.other_annotator = Annotator(id=2, seed=24, num_classes=3)

    def test_init_cm_prob(self):
        cm_prob = self.annotator.init_cm_prob()
        self.assertEqual(cm_prob.shape, (3, 3))
        for row in cm_prob:
            self.assertAlmostEqual(np.sum(row), 1.0, places=5)

    def test_answer(self):
        true_label = 1
        answer = self.annotator.answer(true_label)
        self.assertIn(answer, range(self.annotator.num_classes))
        self.assertEqual(self.annotator.current_answer, answer)
        self.assertEqual(self.annotator.cm[true_label, answer], 1)

    def test_update_accuracy(self):
        # Force some known answers
        self.annotator.cm = np.array([[5, 0, 0], [0, 3, 2], [1, 0, 4]])
        self.annotator.update_accuracy()
        expected_accuracy = np.round(np.array([1.0, 0.6, 0.8]), 4)
        np.testing.assert_array_almost_equal(self.annotator.accuracies, expected_accuracy)

    def test_update_all_metrics(self):
        self.annotator.cm = np.array([[5, 0, 0], [0, 3, 2], [1, 0, 4]])
        self.annotator.update_all_metrics()
        self.assertTrue(hasattr(self.annotator, 'avg_accuracy'))
        self.assertTrue(hasattr(self.annotator, 'avg_precision'))
        self.assertTrue(hasattr(self.annotator, 'avg_recall'))
        self.assertTrue(hasattr(self.annotator, 'avg_f1_score'))

    def test_rate_positive(self):
        self.annotator.current_answer = 1
        self.other_annotator.current_answer = 1
        rating = self.annotator.rate(self.other_annotator)
        self.assertEqual(rating, POSITIVE_RATING)
        self.assertEqual(self.other_annotator.rating_scores[1], POSITIVE_RATING)

    def test_rate_negative(self):
        self.annotator.current_answer = 1
        self.other_annotator.current_answer = 2
        rating = self.annotator.rate(self.other_annotator)
        self.assertEqual(rating, NEGATIVE_RATING)
        self.assertEqual(self.other_annotator.rating_scores[2], NEGATIVE_RATING)

    def test_rating_score_value(self):
        self.annotator.rating_scores = [3, 6, 9]
        score = self.annotator.rating_score_value(2, n_annotators=3, labeling_iteration=3)
        expected_score = 9 / (3 * 3)
        self.assertAlmostEqual(score, expected_score)

    def test_update_reputation_per_class(self):
        self.annotator.rating_scores = [3, 6, 9]
        self.annotator.cm = np.array([[5, 0, 0], [0, 3, 2], [1, 0, 4]])
        reputations = self.annotator.update_reputation_per_class(N=3, iteration=3)
        self.assertEqual(len(reputations), 3)

    def test_accuracy_class(self):
        self.annotator.cm = np.array([[5, 0, 0], [0, 3, 2], [1, 0, 4]])
        accuracies = self.annotator.accuracy_class()
        expected = np.round(np.array([1.0, 0.6, 0.8]), 4)
        np.testing.assert_array_almost_equal(accuracies, expected)

    def test_precision_class(self):
        self.annotator.cm = np.array([[5, 0, 0], [0, 3, 2], [1, 0, 4]])
        precisions = self.annotator.precision_class()
        self.assertEqual(len(precisions), 3)

    def test_recall_class(self):
        self.annotator.cm = np.array([[5, 0, 0], [0, 3, 2], [1, 0, 4]])
        recalls = self.annotator.recall_class()
        self.assertEqual(len(recalls), 3)

    def test_f1_score_class(self):
        self.annotator.cm = np.array([[5, 0, 0], [0, 3, 2], [1, 0, 4]])
        f1_scores = self.annotator.f1_score_class()
        self.assertEqual(len(f1_scores), 3)

    def test_repr(self):
        representation = repr(self.annotator)
        self.assertIsInstance(representation, str)
        self.assertIn('Annotator', representation)

if __name__ == '__main__':
    unittest.main()