import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn import base, pipeline, preprocessing as pre
from sklearn.impute import KNNImputer

langs = pd.read_csv('data/languages.csv')
langs_geo = gpd.GeoDataFrame(
    langs.copy(), geometry=gpd.points_from_xy(langs.Longitude, langs.Latitude)
)

features = pd.read_csv('data/parameters.csv')

values = pd.read_csv('data/values.csv')
present_values = pd.crosstab(values.Language_ID, values.Parameter_ID)

codes = pd.read_csv('data/codes.csv')


def density(df):
    return df.sum().sum() / df.count().sum()


def choose_features_and_languages(
    df, density_threshold, n_features_to_drop, n_languages_to_drop, verbose=True
):
    df_sub = df.copy()
    i = 1
    while density(df_sub) < density_threshold:
        features_sub = df_sub.sum().sort_values(ascending=False)
        features_to_drop = features_sub[
            features_sub <= features_sub[-n_features_to_drop]
        ].index
        languages_sub = df_sub.sum(axis=1).sort_values(ascending=False)
        languages_to_drop = languages_sub[
            languages_sub <= languages_sub[-n_languages_to_drop]
        ].index
        df_sub.drop(features_to_drop, axis=1, inplace=True)
        df_sub.drop(languages_to_drop, inplace=True)
        if verbose and (i % 5 == 0 or density(df_sub) >= density_threshold):
            print(f'Iteration {i}: reached density {density(df_sub):.1%}')
        i += 1
    return df_sub


def choose_and_evaluate_features_and_languages(df, *args, **kwargs):
    df = sort_highest_coverage_first(df)
    result = choose_features_and_languages(df, *args, **kwargs)
    if 'verbose' not in kwargs or kwargs['verbose']:
        num_languages, num_features = result.shape
        print(f'Kept {num_languages} languages and {num_features} features')
        naive_density = density(df.iloc[:num_languages, :num_features])
        print(f"The naive approach would have a density of {naive_density:.1%}")
    return result


def sort_highest_coverage_first(df):
    languages_by_coverage = present_values.sum(axis=1).sort_values(ascending=False)
    features_by_coverage = present_values.sum().sort_values(ascending=False)
    return df.reindex(
        index=languages_by_coverage.index,
        columns=features_by_coverage.index,
    )


present_values_sorted = sort_highest_coverage_first(present_values)


class Sample:
    def __init__(self, present_values, impute=False):
        self.present_values = present_values
        self.langs_list = list(self.present_values.index)
        self.langs = langs_geo[langs_geo.ID.isin(self.langs_list)]
        self.lang_names = list(self.langs.Name)
        self.features_list = list(
            sorted(
                self.present_values.columns,
                key=lambda x: (int(x[:-1]), x[-1:])
            )
        )
        self.features = features[features.ID.isin(self.features_list)]
        self.feature_names = list(self.features.Name)
        self.values = values[
            values.Language_ID.isin(self.langs.ID) &
            values.Parameter_ID.isin(self.features.ID)
        ]
        self.codes = codes[
            codes.Parameter_ID.isin(self.features.ID)
        ]
        self.values_matrix = pd.crosstab(
            self.values.Language_ID,
            self.values.Parameter_ID,
            values=self.values.Value,
            aggfunc='sum',
        ).fillna(-1).astype(int).reindex(
            index=self.langs_list,
            columns=self.features_list,
        )
        if impute:
            self.encoder = PandasColumnTransformer(feature_treatment)
            self.values_encoded = self.encoder.fit_transform(self.values_matrix)
            self.scaler = pipeline.Pipeline([
                ('missing', pre.FunctionTransformer(to_float)),
                ('scaler', pre.MinMaxScaler((0, 1))),
            ])
            self.values_scaled = pd.DataFrame(
                self.scaler.fit_transform(self.values_encoded),
                columns=self.values_encoded.columns,
                index=self.values_encoded.index,
            )
            self.imputer = KNNImputer(weights='distance')
            self.values_scaled_imputed = pd.DataFrame(
                self.imputer.fit_transform(self.values_scaled),
                columns=self.values_encoded.columns,
                index=self.values_encoded.index,
            )
    
    def fcount(self, feature_id):
        """How many languages in the sample have this feature defined?"""
        return self.present_values[feature_id].sum()
    
    def value_names(self, feature_id):
        """What do the numerical value codes represent?"""
        return self.codes[self.codes.Parameter_ID == feature_id][['Name', 'Number']].set_index('Number')
    
    def drop_redundant(self):
        """
        Returns a sample like this one but with the redundant features removed.
        
        There are two groups of redundant features: the word order group and the
        negation group.
        
        The redundant word order features are 95A, 96A, and 97A, which are crosstabs
        between verb-object word order and other features. We can recreate these
        (and any other crosstabs we want) using Pandas, so there's no need
        to keep these features.
        
        The redundant negation features are 143E and 143F, which are recoverable
        from the more comprehensive classification in 143A.
        """
        return Sample(
            self.present_values.drop(['95A', '96A', '97A', '143E', '143F'], axis=1, errors='ignore'),
            impute=True
        )
        

def sample_of_density(density_threshold):
    return Sample(
        choose_and_evaluate_features_and_languages(
            present_values,
            density_threshold=density_threshold,
            n_features_to_drop=1,
            n_languages_to_drop=2,
            verbose=False,
        )
    )


class Ordinal(base.TransformerMixin):
    """
    This variable needs to be kept ordinal, possibly with some values recoded.
    
    The argument, if given, must be a mapping from integers to integers, each
    item indicating that the key should be recoded as the value. For example,
    Ordinal({4: 1}) will replace all 4's with 1's but otherwise leave the
    numbers unchanged.
    """
    
    def __init__(self, recode=None):
        if recode is None:
            recode = {}
        self.recode = recode
    
    def fit(self, x, y=None):
        self._feature_names = list(x.columns)
        return self
    
    def transform(self, x):
        return x.replace(self.recode)
    
    def get_feature_names(self):
        return self._feature_names


class OneHot(base.TransformerMixin):
    """
    This variable needs to be one-hot encoded.
    
    This expects the input to already be pseudo-ordinal, i.e. with
    values ranging from 1 to n. Missing values are indicated by
    -1, and propagated to all one-hot columns.
    
    The argument, if given, must be a mapping from integers to lists
    of integers. Each item indicates that the key, instead of being
    given its own column, must be rewritten as ones in the columns
    given by the values. For example, OneHot({1: [], 4: [2, 3]})
    will give all values except 1 and 4 their own columns, then
    represent 1 as all zeros and 4 as a one in the 2 and 3 columns.
    """
    
    def __init__(self, n, recode=None):
        if recode is None:
            recode = {}
        self.n = n
        self.recode = recode
    
    def fit(self, x, y=None):
        self._removed_values = set(self.recode.keys())
        removed_not_restored_values = self._removed_values - set(
            value for values in self.recode.values()
            for value in values
        )
        self.new_cols = [
            (col, value)
            for col in x.columns
            for value in range(1, self.n + 1)
            if value not in removed_not_restored_values
        ]
        return self
    
    def transform(self, x):
        result = pd.DataFrame()
        for col, value in self.new_cols:
            col_name = f'{col}_{value}'
            if value in self._removed_values:
                result[col_name] = 0
            else:
                result[col_name] = (x[col] == value).astype(int)
        for col in x.columns:
            for key, values in self.recode.items():
                for value in values:
                    col_name = f'{col}_{value}'
                    result[col_name] = result[col_name] | (x[col] == key).astype(int)
        for col, value in self.new_cols:
            col_name = f'{col}_{value}'
            result[col_name] = result[col_name].mask(x[col] == -1, -1)
        return result
    
    def get_feature_names(self):
        return [f'{col}_{value}' for col, value in self.new_cols]


class PandasColumnTransformer(base.TransformerMixin):
    def __init__(self, transformers):
        self.transformers = transformers
    
    def fit(self, x, y=None):
        for col, transformer in self.transformers.items():
            if col in x:
                transformer.fit(x[[col]], y)
        return self
    
    def transform(self, x):
        result = []
        for col, transformer in self.transformers.items():
            if col in x:
                result.append(transformer.transform(x[[col]]))
        return pd.concat(result, axis=1)

    
def to_float(df):
    return df.astype(float).replace(-1.0, np.nan)


feature_treatment = {
    '1A': Ordinal(),
    '2A': Ordinal(),
    '3A': Ordinal(),
    '4A': OneHot(4, {1: [], 4: [2, 3]}),
    '5A': OneHot(5, {2: [], 5: [3, 4]}),
    '6A': OneHot(4, {1: [], 4: [2, 3]}),
    '7A': OneHot(8, {1: [], 5: [2, 3], 6: [2, 4], 7: [3, 4], 8: [2, 3, 4]}),
    '8A': OneHot(5, {1: [], 3: [], 4: [2, 5]}),
    '11A': OneHot(4, {1: [], 2: [3, 4]}),
    '12A': Ordinal(),
    '13A': Ordinal(),
    '18A': OneHot(6, {1: [], 5: [2, 4], 6: [3, 4]}),
    '19A': OneHot(7, {1: [], 6: [2, 4, 5], 7: [4, 5]}),
    '26A': Ordinal({1: 4}),
    '33A': OneHot(9, {9: []}),
    '51A': OneHot(9, {9: []}),
    '57A': OneHot(4, {3: [1, 2], 4: []}),
    '69A': OneHot(5, {5: []}),
    '70A': OneHot(5, {1: [1, 2, 3], 2: [1, 2], 3: [1, 3], 4: [2, 3], 5: []}),
    '81A': OneHot(7),
    '82A': OneHot(3),
    '83A': OneHot(3),
    '85A': OneHot(5, {5: []}),
    '86A': OneHot(3),
    '87A': OneHot(4),
    '88A': OneHot(6),
    '89A': OneHot(4),
    '90A': OneHot(7),
    '92A': OneHot(6, {6: []}),
    '93A': OneHot(3),
    '112A': OneHot(6, {5: [1, 2]}),
    '116A': OneHot(7, {3: [1, 2], 7: []}),
    '143A': OneHot(17, {
        6: [1, 2], 7: [1, 3], 8: [1, 4], 9: [2, 3], 10: [2, 4], 11: [3, 4], 12: [3],
        13: [6], 14: [6], 15: [6], 16: [6], 17: [6],
    }),
    '143G': OneHot(4, {4: []}),
    '144A': OneHot(21, {
        2: [3], 3: [4], 4: [2], 5: [1], 6: [3], 7: [4], 8: [2],
        9: [1], 10: [4], 11: [2], 12: [1], 13: [3], 14: [4], 15: [2],
        16: [5], 17: [5], 18: [5], 19: [5], 20: [5], 21: [5],
    }),
}


s229 = sample_of_density(0.98)
s229d = s229.drop_redundant()

s280 = sample_of_density(0.95)
s280d = s280.drop_redundant()