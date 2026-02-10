# Studying ML Model Reliability under Covariate Shift
### Reproduction and Extension of:
"A Survey on Dataset Shift in Machine Learning"
Quinonero-Candela et al., 2009

Machine learning models usually expect that the data they see during training looks the same as what they'll see when put to the test. But in real life, that just doesn’t hold up—things change, and the data shifts.

In this experiment, we dig into three things:
1. What happens to model performance when the data starts to drift
2. How to spot that drift with statistical tests
3. How systems can bounce back automatically by retraining

Drift Simulation → Model Degradation → Drift Detection → Drift Decision → Retraining → Recovery

- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- KNN

- KS Test
- PSI
- Mean Shift
As drift increases, accuracy drops significantly for all models.
However, tree-based models degrade slower than linear models.

![Accuracy vs Drift](plots/accuracy_vs_drift.png.png)

![Drift Detection](plots/drift_dectection.png.png)

As drift increases, accuracy drops significantly for all models.
However, tree-based models degrade slower than linear models.
When drift is detected, the model is retrained on new data,
restoring performance close to original levels.

• ML models are highly sensitive to covariate shift.
• KS and PSI effectively detect distribution change before accuracy collapses.
• Retraining restores performance, demonstrating need for monitoring in ML systems.
• Different models show different robustness patterns under drift.
This study matters if you’re working with IoT sensors that move around or if you’ve got learning-to-rank systems where the kinds of queries keep shifting. It’s also important for machine learning setups running out in the real world, especially when things don’t stay the same for long. And if you care about keeping your AI or ML monitoring solid and trustworthy, this is for you, too.

This project shows how to build a full machine learning reliability pipeline—one that actually helps you spot data drift, figure out what’s going on, and get things back on track when problems pop up.

Here, I’m recreating the covariate shift experiment from the paper. Basically, the idea is simple: you train your model on one distribution, 
then test it on data that comes from a different one. The result? The model just doesn’t perform as well. This setup exposes how sensitive machine learning models are when the world changes between training and testing.
Paper Reproduction: Dataset Shift and Covariate Shift

Paper: "A Survey on Dataset Shift in Machine Learning"  
Authors: Joaquin Quiñonero-Candela, Masashi Sugiyama, Anton Schwaighofer, Neil D. Lawrence  
Year: 2009

For this project, I set out to recreate some of the core ideas from this paper by running hands-on experiments that show what happens when a machine learning model faces data that doesn’t match what it saw during training.

So, what’s the problem here? The paper digs into something called dataset shift—a big headache in machine learning. Basically, you train your model on one batch of data, but when you deploy it, suddenly the data looks different. The model gets confused and starts making mistakes.

The paper breaks dataset shift into a few types:

- Covariate Shift: The input features change, but the way inputs map to outputs stays the same.
- Prior Probability Shift: The balance between classes changes.
- Concept Shift: The relationship between the inputs and outputs actually changes.

Most ML models just assume the training and test data will look the same. In reality, that almost never happens. The paper really hammers that point.

What Did I Do?

I focused on covariate shift. Using the Bank Marketing dataset, I tweaked features like age, nr.employed, duration, and campaign on purpose to mess with the data distribution.

Here’s how my project lines up with the paper:

- Covariate Shift: I changed key input features artificially.
- Watching the Model Fail: I tracked accuracy as drift increased, then plotted it.
- Silent Model Failure: I compared confusion matrices—one with no drift, one with lots of drift—to see how the errors crept in.
- Drift Detection: I ran Deepchecks FeatureDrift and LabelDrift to spot issues.
- Real-world Simulation: I pulled out data subsets (like age > 60, duration > 500) to mimic extreme drift.

What Did I Find?

- Accuracy drops as the data drifts further away from what the model saw during training.
- Drift detection tools actually catch these changes before things get too bad.
- The model’s predictions can swing wildly when covariate shift hits.
- Graphs from the project (accuracy_vs_drift, drift_detection, confusion matrices) lay this out clearly.

Some Takeaways

- You can’t trust accuracy alone when dataset shift sneaks in.
- Models might look fine at first, but they can fail quietly—no obvious red flags.
- If you watch feature drift closely, you can catch problems early.
- Tracking your data is just as important as building the model itself.
- Certain slices of real-world data can feel like extreme drift scenarios.

Why Does This Matter?

This whole experiment makes it obvious why people care so much about making ML models robust and able to generalize. If you ignore dataset shift, your system just won’t hold up in the real world.

Wrapping Up

This project is a practical take on the dataset shift theory from the paper. By running real experiments, I showed how covariate shift messes with models and how you can actually detect and study this drift before it causes too much trouble.
