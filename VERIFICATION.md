# Verification record

Verified locally on 18 October 2026 with Python 3.12.14, NumPy 2.3.5 and Pillow 12.3.0.

- `python -m unittest discover -s tests -v`: all **6 tests passed**.
- `python -m perception_lab.demo --output results`: completed both model-training runs, saved numeric checkpoints, scene manifests, learning histories, confusion metrics and PNG figures.
- `python -m perception_lab.predict results/example_camera.png --model results/style_augmented_classifier.npz --output results/example_prediction.png`: completed single-image inference successfully.
- Comparison and training-curve figures were visually inspected for readable labels and layout.

Recorded results: baseline standard / shifted mean IoU **0.9999 / 0.1176**; model with both training styles standard / shifted mean IoU **0.8763 / 0.8868**.

This verifies reproducible execution and the reported synthetic experiment. It does not verify real-camera performance, safe traversability, ROS integration or hardware robotics.
