# File list

* `Autoencoder` - Can autoencoders effectively reduce SE data to their true intrinsic dimensionality? 
  * **Yes**, but they may not be the best approach.
* `Biased SVMs with weighted fuzzy oversampling` - Can we use WFO to bias SVMs towards better recalls and lower pfs?
  * **Yes**, with varying levels of success. Works better on deep learners, see the GHOST paper.
* `CFS` - A test of correlation-based feature sampling.
* `Combining CSVs for issue close time prediction` - Combines GitHub issue close time datasets to one, multi-class dataset.
  * **No**, don't get better results.
* `Committee of models` - A committee of models to vote (like an ensemble)
  * **No**.
* `Defect prediction` - Some defect prediction tinkering.
* `Knowledge distillation` - KD trial on defect prediction dataset
  * **Yes**, but may need better KD approaches.
* `Linear regions in NNs` - Demonstration that more linear regions not necessarily better.
* `Neural tangent kernel + SMOTE for defect prediction` - Basically the title.
  * **No**, does not work.
* `NTK` - Tinkering with the neural tangent kernel.
* `Outlier removal test` - Learn only from inliners
  * **Yes**, with some degree of success.
* `Rolles theorem` - A hyper-parameter tuner based off Rolle's theorem.
  * **No**; works in very specific cases ("sufficiently continuous" hyper-param spaces, sufficiently capable learners)
* `Untitled` - Barely started, a cooperative network system for learning data augmentation + task
* `Weighted losses` - Weighted loss functions.
  * **Yes**, see GHOST paper.