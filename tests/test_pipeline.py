import tempfile
import unittest
from pathlib import Path
import numpy as np
from perception_lab.data import features, make_splits, sample_pixels, scene
from perception_lab.metrics import confusion_matrix, summarize
from perception_lab.model import PixelClassifier


class DataTests(unittest.TestCase):
    def test_scene_ids_and_seeds_are_disjoint_before_pixel_sampling(self):
        splits = make_splits()
        ids, seeds, signatures = set(), set(), set()
        for scenes in splits.values():
            for item in scenes:
                self.assertNotIn(item.scene_id, ids)
                self.assertNotIn(item.seed, seeds)
                signature = item.image.tobytes()
                self.assertNotIn(signature, signatures)
                ids.add(item.scene_id)
                seeds.add(item.seed)
                signatures.add(signature)
        self.assertEqual(len(ids), 260)

    def test_scene_is_reproducible_and_has_valid_labels(self):
        first, second = scene(101), scene(101)
        np.testing.assert_array_equal(first.image, second.image)
        np.testing.assert_array_equal(first.labels, second.labels)
        self.assertEqual(set(np.unique(first.labels)), {0, 1, 2})
        self.assertEqual(features(first.image).shape, (64 * 96, 21))
        self.assertTrue(np.isfinite(features(first.image)).all())


class MetricTests(unittest.TestCase):
    def test_hand_calculated_iou_and_floor_precision(self):
        # True rows / predicted columns: IoUs = 2/4, 3/5, 4/6.
        matrix = np.array([[2, 1, 0], [0, 3, 1], [1, 0, 4]])
        result = summarize(matrix)
        self.assertAlmostEqual(result["mean_iou"], (2/4 + 3/5 + 4/6) / 3)
        self.assertAlmostEqual(result["pixel_accuracy"], 9/12)
        self.assertAlmostEqual(result["floor_precision"], 3/4)
        self.assertAlmostEqual(result["floor_recall"], 3/4)

    def test_perfect_wrong_absent_and_invalid_inputs(self):
        truth = np.array([0, 1, 2, 1])
        perfect = summarize(confusion_matrix(truth, truth))
        self.assertEqual(perfect["mean_iou"], 1.0)
        wrong = summarize(confusion_matrix(truth, (truth+1) % 3))
        self.assertEqual(wrong["mean_iou"], 0.0)
        absent = summarize(confusion_matrix(np.array([0, 0]), np.array([0, 0])))
        self.assertIsNone(absent["iou_by_class"]["floor"])
        self.assertIsNone(absent["floor_precision"])
        for actual, pred in [(np.array([]), np.array([])), (np.array([3]), np.array([0])),
                             (np.array([1]), np.array([0, 1]))]:
            with self.assertRaises(ValueError):
                confusion_matrix(actual, pred)


class ModelTests(unittest.TestCase):
    def test_learning_train_only_normalization_serialization_and_inference(self):
        train = [scene(10), scene(11), scene(12)]
        x, y = sample_pixels(train, per_class=60)
        validation = scene(21)
        vx, vy = features(validation.image), validation.labels.ravel()
        classifier = PixelClassifier()
        history = classifier.fit(x, y, vx, vy, epochs=8, batch_size=128)
        np.testing.assert_allclose(classifier.mean, x.mean(axis=0))
        self.assertGreater(history[0]["training_cross_entropy"], history[-1]["training_cross_entropy"])
        score = summarize(confusion_matrix(vy, classifier.predict(validation.image)))
        for key in ("mean_iou", "pixel_accuracy", "floor_precision", "floor_recall"):
            self.assertTrue(0 <= score[key] <= 1)
        # Keep test scratch data in the project, including sandboxed environments.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "model.npz"
            classifier.save(path)
            restored = PixelClassifier.load(path)
            np.testing.assert_array_equal(classifier.predict(validation.image), restored.predict(validation.image))
        self.assertEqual(classifier.predict(validation.image).shape, (64, 96))

    def test_inference_before_training_fails(self):
        with self.assertRaises(RuntimeError):
            PixelClassifier().predict(scene(20).image)


if __name__ == "__main__":
    unittest.main()
