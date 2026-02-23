ELEC2 Drift Study — How Well Does the KS Test Really Work on Real Data?

What’s Going On Here

Most drift detection studies mess with data on purpose—add some noise, shift some averages, that sort of thing. But life’s messier than that. Here, I’m looking at the ELEC2 electricity dataset: real electricity prices from New South Wales, Australia, where the data naturally drifts over time. The big question: does the Kolmogorov-Smirnov (KS) test actually warn us when model accuracy drops in the real world, or does it cry wolf?

About the Data

The ELEC2 dataset tracks whether the electricity price went up or down compared to the previous day’s average. You’ve got 45,312 samples covering two years, so it’s not tiny. What’s great about it? The distribution actually changes over time—thanks to real market shifts—so we don’t have to fake anything.

How I Ran This

I split the data into five chunks, each covering a stretch of time, and trained a logistic regression model on the very first chunk. No retraining after that. The model gets tested on all five windows, just like what happens when you deploy a model in the wild and let it age.

For each window, I measured two things: how accurate the model was and the mean KS score (averaged across all features) compared to the original chunk. Then I checked—when the KS score goes up, does accuracy drop? And do they always move together?

What Happened

Window | Accuracy | Mean KS Score  
0 (Reference) | 0.820 | 0.000  
1 | 0.805 | 0.093  
2 | 0.665 | 0.444  
3 | 0.645 | 0.545  
4 | 0.820 | 0.360  

Pearson correlation (r) between KS score and accuracy drop: 0.807

Most of the time, the KS statistic tracks model accuracy pretty well. As drift grows, the model struggles. But look at Window 4: the KS score is high (0.36—way past the usual warning threshold), but accuracy jumps right back to its original level. So the KS test isn’t telling the full story. It flags both long-lasting drift and short-term blips, but only the first one really hurts your model in the end.

The Main Takeaway

KS-based drift detection works well when the drift sticks around. But if the distribution just wanders off for a bit and comes back, you get a false alarm. For electricity markets (and other systems that cycle), this means you could end up retraining your model for no good reason, wasting time and compute.

Graphs

- accuracy_over_time.png: Model accuracy across all five windows
- ks_drift_over_time.png: Mean KS score over time
- ks_vs_accuracy_drop.png: KS score vs. accuracy drop (correlation: 0.807)
- confusion_matrices.png: How prediction errors shift with drift
- retraining_recovery.png: Accuracy jumps back after retraining

What’s Missing and What’s Next

I only used logistic regression and the KS test here. Next up: compare with PSI and MMD, try other models, and see if tracking how long drift lasts (not just how big it is) helps us avoid these false positives.

Author

Mrinank — BS Data Science, IIT Madras (2nd Year)
