"""
Tools for modelling features based on language family and geographical area.
"""

import math
import textwrap

import numpy as np
import pandas as pd

from sklearn import preprocessing as pre, pipeline
from sklearn import model_selection as ms, metrics
from sklearn import linear_model as lm

import walsdata


class OriginDataset:
    """
    Data on language features, family and geographical area.
    
    Provides methods to train and evaluate different models on the
    data.
    
    Parameters:
    - values: A table of each language's value for each language feature
    - origins: A table giving the family and area of each language
    - categories (list of lists): The list of acceptable family/region
      categories in each column of origins
    - random_state: The random state to use for the train-test split
    """
    def __init__(self, values, origins, categories, random_state=5312):
        self.values = values.reindex(origins.index)
        self.origins = origins
        self.categories = categories
        
        self.values_train, self.values_test, self.origins_train, self.origins_test = (
            ms.train_test_split(self.values, self.origins, random_state=random_state)
        )

    def origins_onehot(self, clf):
        onehot = pre.OneHotEncoder(
            categories=self.categories,
            handle_unknown='ignore',
        )

        model = pipeline.Pipeline([
            ('onehot', onehot),
            ('clf', clf),
        ])
        return model
    
    def logistic_model(self, feature, cv=False, random_state=5364):
        """
        Logistic regression on one feature
        """
        if cv:
            logreg = lm.LogisticRegressionCV(
                random_state=random_state, scoring='neg_log_loss',
                Cs=[0.2, 0.3, 0.5, 0.8, 1, 1.5, 2]
            )
        else:
            logreg = lm.LogisticRegression(random_state=random_state)
        
        model = self.origins_onehot(logreg)
        
        train = round_to_int(self.values_train[feature])
        test = round_to_int(self.values_test[feature])
        full = round_to_int(self.values[feature])
        
        model.fit(self.origins_train, train)

        clf = model.named_steps['clf']

        train_score = log_odds_vs_baseline(
            train, model.predict_proba(self.origins_train)
        )
        test_score = log_odds_vs_baseline(
            test, model.predict_proba(self.origins_test)
        )
        coefs = clf.coef_
        neutral_prob = clf.predict_proba(np.full(clf.coef_.shape, 0.0))[0, 1]
        observed_prob = full[full == 1].count() / full.count()

        feature_name = walsdata.get_shortname(feature)
        named_coefs = self.named_coefs(coefs[0])

        result = OriginResults(
            feature_name=feature_name,
            train_score=train_score,
            test_score=test_score,
            observed_prob=observed_prob,
            innate_prob=neutral_prob,
            coefs=named_coefs,
        )

        return result
    
    def full_logistic_model(self, cv=False, random_state=5364):
        cat_features = list(self.values.columns[self.values.columns.str.contains('_')])
        result = []
        for feature in cat_features:
            try:
                result.append(
                    (feature, self.logistic_model(feature, cv=cv, random_state=random_state))
                )
            except ValueError:
                # Feature missing from the training set
                pass
        return pandify(result)
    
    def full_linear_model(self, cv=False, random_state=5364):
        ord_features = list(self.values.columns[~self.values.columns.str.contains('_')])
        
        if cv:
            linreg = lm.RidgeCV(
                cv=5,
                alphas=(0.01, 0.1, 1, 10, 100)
            )
        else:
            linreg = lm.Ridge(random_state=random_state)

        model = self.origins_onehot(linreg)

        train = self.values_train[ord_features]
        test = self.values_test[ord_features]
        full = self.values[ord_features]

        model.fit(self.origins_train, train)

        clf = model.named_steps['clf']

        train_score = metrics.r2_score(
            train, model.predict(self.origins_train), multioutput='raw_values'
        )
        test_score = metrics.r2_score(
            test, model.predict(self.origins_test), multioutput='raw_values'
        )
        coefs = clf.coef_
        neutral_value = clf.predict(np.full((1, clf.coef_.shape[1]), 0.0))
        observed_value = full.mean().values
        
        result = []
        for i, feature in enumerate(ord_features):
            result.append((
                feature,
                OriginResults(
                    feature_name=walsdata.get_shortname(feature),
                    train_score=train_score[i],
                    test_score=test_score[i],
                    observed_prob=observed_value[i],
                    innate_prob=neutral_value[0, i],
                    coefs=self.named_coefs(clf.coef_[i]),
                )
            ))
        return pandify(result)
    
    def named_coefs(self, coefs):
        return dict(
            zip((cat for col_cat in self.categories for cat in col_cat), coefs)
        )


def round_to_int(series):
    """
    Round a float series to the nearest integer and coerce to int
    """
    return series.round().astype(int)


def log_odds_vs_baseline(y_true, y_pred):
    """
    Custom metric for evaluating binary probability predictions.
    
    Like R^2, it goes from -infinity to 1, where a perfect
    model gets a score of 1, a baseline model gets a score
    of 0, and a maximally terrible model (one that assigns
    a probability of 0 to all the correct labels) gets a score of
    -infinity. Good models get some positive score less
    than one.
    
    This is computed as 1 - LL / LLB, where LL is the log
    loss and LLB is the log loss of the baseline model.
    """
    baseline_model = np.ones(y_true.shape) * np.sum(y_true, axis=0) / len(y_true)
    baseline_log_loss = metrics.log_loss(y_true, baseline_model)
    log_loss = metrics.log_loss(y_true, y_pred)
    return 1 - log_loss / baseline_log_loss


def pandify(origin_results):
    features, origin_results = zip(*origin_results)
    df = pd.DataFrame(index=map(walsdata.get_shortname, features))
    df['training_score'] = [result.train_score for result in origin_results]
    df['testing_score'] = [result.test_score for result in origin_results]
    df['observed_rate'] = [result.observed_prob for result in origin_results]
    df['innate_rate'] = [result.innate_prob for result in origin_results]
    df['odds_shift'] = odds_shift(df['innate_rate'], df['observed_rate'])
    df['log_odds_shift'] = np.log(df['odds_shift'])
    for coef in origin_results[0].coefs:
        df[coef] = [result.coefs[coef] for result in origin_results]
    return df


def odds_shift(p0, p):
    return odds(p) / odds(p0)


def odds(p):
    return p / (1 - p)


class OriginResults:
    def __init__(
        self,
        feature_name,
        train_score,
        test_score,
        observed_prob,
        innate_prob,
        coefs,
    ):
        self.feature_name = feature_name
        self.train_score = train_score
        self.test_score = test_score
        self.observed_prob = observed_prob
        self.innate_prob = innate_prob
        self.coefs = coefs
    
    def __str__(self):
        max_coef_name_len = max(len(coef_name) for coef_name in self.coefs)
        coef_display = '\n        '.join([
                f'{coef_name.ljust(max_coef_name_len)}: {coef_value:.3g}'
                for coef_name, coef_value in self.coefs.items()
        ])
        return textwrap.dedent(
            f"""\
            Feature {self.feature_name}:
                Training score: {self.train_score:.3g}
                Testing score: {self.test_score:.3g}
                Observed rate: {self.observed_prob:.3g}
                Inferred innate rate: {self.innate_prob:.3g}
                Coefficients:
            """
        ) + '        ' + coef_display
    
    def __repr__(self):
        return str(self)
