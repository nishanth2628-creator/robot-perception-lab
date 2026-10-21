# Understand and extend this project

This is a study guide for an AI-assisted project, not a script for claiming work you have not understood. Run the experiment and trace one image from generation to its confusion matrix before adding the project to an application.

## Questions you should be able to answer

1. **What problem does the project solve?** It investigates how a lightweight learned pixel classifier behaves on synthetic camera scenes when appearance changes. It does not establish real-world obstacle avoidance.
2. **Why is the first test result almost perfect?** The standard renderer has easy, consistent colour cues. New random scenes from the same renderer still share those shortcuts.
3. **What exactly is learned?** A 22 × 3 weight matrix: 21 standardized inputs plus an intercept for three classes. Feature design is manual; class weights are optimized from labelled pixels.
4. **What does the model output?** Three softmax scores per pixel and the class with the largest score. The scores have not been calibrated as reliable probabilities.
5. **What loss and optimization are used?** Average cross entropy with L2 regularization, minimized using mini-batch Adam. The bias is excluded from the L2 penalty.
6. **Why split by scene rather than pixel?** Random pixel splits would let near-identical textures and geometry from the same image appear in both training and evaluation, inflating the measured result.
7. **How is preprocessing kept separate?** Feature means and standard deviations are computed only on the model's training pixels. Validation and test features use those saved values.
8. **How is the epoch selected?** Mean IoU on the appropriate validation scenes. The baseline selects epoch 5 and the model trained on both styles selects epoch 17 in the recorded run.
9. **Why use IoU as well as accuracy?** Large wall/floor regions can dominate accuracy. Each class IoU compares intersection with union; averaging the class IoUs gives obstacles more visibility.
10. **What did adding the shifted style change?** Shifted-test mean IoU improved from 0.1176 to 0.8868; standard-test mean IoU decreased from 0.9999 to 0.8763. Both results matter.
11. **Is the improved result an unseen-domain result?** No. The shifted appearance style is represented by separate training and validation scenes. Only the final scene seeds are held out.
12. **Why is floor precision relevant to robotics?** A false floor prediction may hide an obstacle. The improved shifted test still labels 5,518 obstacle pixels as floor, so it cannot be used as a navigation safety system.
13. **What would real deployment need?** Representative real images, camera calibration and geometry, clearance/depth reasoning, appropriate safety controls, temporal evaluation and a separate deployment validation process.
14. **What contribution can you honestly describe?** Say it is an AI-assisted educational project. Explain what you reproduced, investigated and personally changed. Do not claim independent implementation or real-robot testing unless you have done it.

## Three extensions to make your own

- Change the renderer so obstacles sometimes share the exact floor colour. Predict which features will fail, run the experiment and explain the measured result.
- Implement a median filter or a spatial feature, compare it with the current masks, and check whether small obstacles disappear.
- Add 30 untouched scenes with a new appearance rule (for example blue floor and patterned obstacles). Freeze model selection before evaluating that third style. Report every class IoU, including failures.

Keep a short experiment journal with your hypothesis, the precise change, the seed split and the result. That evidence is more useful in an interview than an unexplained score.
