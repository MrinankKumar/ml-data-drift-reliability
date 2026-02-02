Performance Degradation of Machine Learning Models Under Feature Drift and Covariate Shift

1. Motivation

Machine learning models often perform well on test data, but they can struggle in real-life settings due to changes in data distribution over time. This issue is called distribution shift or data drift. Understanding how models react to this drift is essential for creating reliable AI systems.

This project looks into how a Logistic Regression model’s performance declines when feature distributions are changed to mimic real-world drift situations.

2. Dataset

We used the Bank Marketing Dataset. The goal is to predict whether a customer will subscribe to a term deposit (y).

Features such as:

- age
- duration
- campaign
- nr.employed

were used to imitate realistic drift conditions.

3. Baseline Model

We trained a Logistic Regression model with StandardScaler using a Scikit-learn Pipeline. This setup represents a common production ML pipeline.

We measured baseline accuracy on clean test data.

4. Simulating Drift

We artificially introduced different levels of drift:

Drift Level | Method
--- | ---
Low | Small noise added to age
Medium | Noise added to age and nr.employed
High | Feature scaling and permutation
Extreme | Heavy noise across multiple features

This simulates how data changes in real deployment environments.

5. Observations

As the intensity of drift increased, model accuracy went down consistently.

This shows that:

Even a strong model can fail quietly when the input data distribution changes.

6. Deepchecks Validation

We used Deepchecks to:

- Detect Feature Drift
- Detect Label Drift
- Perform Data Integrity checks

The tool correctly identified drifted features, confirming the experimental setup.

7. Why StandardScaler Helps

An experiment without StandardScaler showed greater performance decline. This proved that preprocessing is important for maintaining robustness.

8. Realistic Drift Scenario

Instead of using artificial noise, we analyzed subsets of real data:

- age > 60
- duration > 500
- nr.employed > 2000

These represent real population shifts and resulted in a significant drop in accuracy.

9. Key Insight

This project highlights that simply evaluating accuracy on test data isn’t enough for deployable AI. Models need testing under conditions of distribution shift.

10. Conclusion

This study presents practical methods to assess the robustness and generalization of ML models. Such evaluation is crucial before deploying models in real-world applications.
