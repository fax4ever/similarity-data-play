from collections import defaultdict
from time import time
from sklearn import metrics
import numpy as np

class Evaluation:
    def __init__(self, labels):
        self.evaluations = []
        self.evaluations_std = []
        self.labels = labels

    def fit_and_evaluate(self, km, X, name=None, n_runs=5):
        name = km.__class__.__name__ if name is None else name

        train_times = []
        scores = defaultdict(list)
        for seed in range(n_runs):
            km.set_params(random_state=seed)
            t0 = time()
            km.fit(X)
            train_times.append(time() - t0)
            scores["Homogeneity"].append(metrics.homogeneity_score(self.labels, km.labels_))
            scores["Completeness"].append(metrics.completeness_score(self.labels, km.labels_))
            scores["V-measure"].append(metrics.v_measure_score(self.labels, km.labels_))
            scores["Adjusted Rand-Index"].append(
                metrics.adjusted_rand_score(self.labels, km.labels_)
            )
            scores["Silhouette Coefficient"].append(
                metrics.silhouette_score(X, km.labels_, sample_size=2000)
            )
        train_times = np.asarray(train_times)

        print(f"clustering done in {train_times.mean():.2f} ± {train_times.std():.2f} s ")
        evaluation = {
            "estimator": name,
            "train_time": train_times.mean(),
        }
        evaluation_std = {
            "estimator": name,
            "train_time": train_times.std(),
        }
        for score_name, score_values in scores.items():
            mean_score, std_score = np.mean(score_values), np.std(score_values)
            print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")
            evaluation[score_name] = mean_score
            evaluation_std[score_name] = std_score
        self.evaluations.append(evaluation)
        self.evaluations_std.append(evaluation_std)