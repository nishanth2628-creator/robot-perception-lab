# Model card: synthetic pixel classifier

**Intended use:** education, reproducible portfolio experiments, and examining appearance shortcuts in a simple segmentation model.

**Out of scope:** physical robot control, collision avoidance, real-camera accuracy claims, or decisions involving people or safety.

**Models:** multinomial logistic regression with 21 hand-crafted RGB/spatial/edge features plus an intercept, three output classes, training-only feature normalization, cross entropy and mini-batch Adam. Training uses at most 320 pixels per class per scene.

**Data:** procedurally generated 96 × 64 indoor-style images with exact masks. Standard scenes have bright walls, warmer floors and saturated boxes. Shifted scenes have darker surfaces and pale boxes. All data is synthetic; there is no external dataset licence or personal data.

**Two checkpoints:** `results/pixel_classifier.npz` trains on 80 standard scenes. `results/style_augmented_classifier.npz` trains on 80 standard plus 80 shifted-style scenes. Validation scenes choose the epoch; 30 standard and 30 shifted test scenes remain held out for evaluation. The second checkpoint has seen the shifted style during training and must not be described as an unseen-domain solution.

**Recorded aggregate mean IoU:** baseline standard 0.9999 / shifted 0.1176; both-style model standard 0.8763 / shifted 0.8868. Per-class results, confusion matrices and per-scene metrics are in `results/metrics.json`. The aggregate metrics do not quantify uncertainty or real-world performance.

**Failure modes:** colours, camera position and geometry strongly influence the output. Unusual lighting, texture, overlapping objects and low contrast are not adequately represented. Floor masks do not represent depth, free clearance or safe traversability. In the shifted evaluation, the both-style model still predicts floor for 5,518 obstacle pixels.

**Reproducibility:** all data seeds, 18-epoch training configuration, sample counts, dependency versions and selected epochs are recorded in the repository. The saved model is a numeric `.npz` file; loading uses `allow_pickle=False`. Tests cover split separation, training normalization, known metrics, learning and inference round trips.

**Provenance:** AI-assisted educational implementation prepared for Nishanth Senthil Kumar, executed locally. No independent human mastery, professional delivery, ROS integration or robot hardware work is implied.
