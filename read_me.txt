The code is written in python. You will also need to be able to run jupyter notebooks.

You will need the following python modules:
numpy
matplotlib
scipy
pandas
sys
os
pickle
sklearn
networkx
keras
seaborn

Explanation of files:
embedding_class.py: class that creates synthetic item features and judgement vector
synthetic_pair_class.py: class for working with synthetic data. Given an object from embedding_class it creates synthetic preference data, computes the bounds from Thm 1, gets prediction error, gets kendall tau correlation with the true ranking and an estimated ranking, gets the estimation error of the true judgement vector with an estimate, gets intransitivity rates,
experiment_utils.py: for running synthetic experiments in the paper
preference_utils.py: This file contains code that is useful for the synthetic_pair_class and the fit_model_class and also the main code
fit_model_class.py:  This class contains code that takes features and pairwise comparisons and fits the salient feature preference model with the top t selection function. It can then fit the features BTL model by setting t = ambient dimension. It also can compute the empirical number of stochastic transitivity violations.
utils.py: helps with making plots in paper
ranknet.py: contains code to train a one hidden layer ranknet model
create_zappos_split.py: this file parses the zappos data and splits into a train/validation/test set
ranknet_zappos.py: runs ranknet experiments for zappos data
zappos_exp.py: runs the MLE with the top-t selection function experiments on the zappos data
synthetic_data_experiments.ipynb: notebook to run synthetic data experiments and create plots

Synthetic data:
Use embedding_class.py and synthetic_pair_class.py to create synthetic data.
Run the 'synthetic_data_experiments.ipynb' notebook in the 'synthetic_experiments' folder for the synthetic data experiments in the paper.

Real datasets:
1. The district data was obtained from the authors of https://gking.harvard.edu/publications/how-measure-legislative-district-compactness-if-you-only-know-it-when-you-see-it. We would be happy to share the code we wrote to parse their data and to run our experiments if requested.
2. The Zappos data can be downloaded here: http://vision.cs.utexas.edu/projects/finegrained/utzap50k/. In particular, we used: ut-zap50k-data.zip (3 MB) and ut-zap50k-feats.zip (208 MB)
To run the zappos experiments after downloading the data,
1. Run create_zappos_split.py to parse the zappos data and create a train/validation/test split
2. Run ranknet_zappos.py for the ranknet experiments
3. Run zappos_exp.py
