import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, balanced_accuracy_score, 
    f1_score, fbeta_score, recall_score, precision_score,
    average_precision_score, confusion_matrix
)
import optuna
from optuna.samplers import TPESampler


class RepeatedNestedCV:
    def __init__(self, estimators, param_spaces, metrics=None, 
                 n_repeats=10, n_outer_folds=5, n_inner_folds=3, 
                 n_trials=5, random_state=42):

        self.estimators = estimators
        self.param_spaces = param_spaces
        self.n_repeats = n_repeats
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.n_trials = n_trials
        self.random_state = random_state

        if metrics is None:
            self.metrics = {
                'MCC': matthews_corrcoef,
                'AUC': roc_auc_score,
                'BA': balanced_accuracy_score,
                'F1': f1_score,
                'F2': lambda y_true, y_pred: fbeta_score(y_true, y_pred, beta=2),
                'Recall': recall_score,
                'Precision': precision_score,
                'PRAUC': average_precision_score,
                'NPV': self.negative_predictive_value,
                'Specificity': self.specificity
            }
        else:
            self.metrics = metrics

        self.results = {
            'estimator': [],
            'repeat': [],
            'outer_fold': [],
            **{metric: [] for metric in self.metrics}
        }

    def negative_predictive_value(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn = cm[0, 0] if cm.shape[0] > 0 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 else 0
        return tn / (tn + fn) if (tn + fn) > 0 else 0.0

    def specificity(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn = cm[0, 0] if cm.shape[0] > 0 else 0
        fp = cm[0, 1] if cm.shape[1] > 1 else 0
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    def _objective(self, trial, estimator, param_space, X, y, cv):
        params = {key: trial.suggest_categorical(key, values) 
                  for key, values in param_space.items()}
        model = clone(estimator).set_params(**params)

        scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_proba)
            scores.append(score)

        return np.mean(scores)

    def fit(self, X, y):
        X = X.copy()
        y = y.copy()
        rng = np.random.RandomState(self.random_state)

        for repeat in range(self.n_repeats):
            repeat_seed = rng.randint(1_000_000)
            outer_cv = StratifiedKFold(
                n_splits=self.n_outer_folds, shuffle=True, 
                random_state=repeat_seed
            )

            for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                inner_cv_seed = rng.randint(1_000_000)
                inner_cv = StratifiedKFold(
                    n_splits=self.n_inner_folds, shuffle=True,
                    random_state=inner_cv_seed
                )

                for name, estimator in self.estimators:
                    space = self.param_spaces[name]
                    study = optuna.create_study(
                        direction='maximize',
                        sampler=TPESampler(seed=self.random_state)
                    )
                    study.optimize(
                        lambda trial: self._objective(
                            trial, estimator, space, X_train, y_train, inner_cv
                        ),
                        n_trials=self.n_trials,
                        n_jobs=1 
                    )

                    best_model = clone(estimator).set_params(**study.best_params)
                    best_model.fit(X_train, y_train)
                    y_pred = best_model.predict(X_test)
                    y_proba = best_model.predict_proba(X_test)[:, 1]

                    self.results['estimator'].append(name)
                    self.results['repeat'].append(repeat)
                    self.results['outer_fold'].append(outer_fold)

                    for metric_name, metric_func in self.metrics.items():
                        if metric_name in ['AUC', 'PRAUC']:
                            score = metric_func(y_test, y_proba)
                        else:
                            score = metric_func(y_test, y_pred)
                        self.results[metric_name].append(score)

        self.results_df = pd.DataFrame(self.results)
        return self

    def get_results(self):
        return self.results_df

    def summarize_results(self):
        if not hasattr(self, 'results_df'):
            raise RuntimeError("Run fit() first.")

        summary = self.results_df.groupby('estimator').agg(
            {metric: ['median', 
                      lambda x: np.percentile(x, 2.5), 
                      lambda x: np.percentile(x, 97.5)]
             for metric in self.metrics}
        )
        summary.columns = ['_'.join([c if isinstance(c, str) else 'CI' for c in col]) for col in summary.columns]
        return summary

    def get_best_estimator(self, metric='MCC'):
        summary = self.summarize_results()
        best = summary[f'{metric}_median'].idxmax()
        return best, summary.loc[best]
