import pandas as pd
import geopandas as gpd

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
        if impute:
            # TODO encode features and impute missing values
            pass
    
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


s229 = sample_of_density(0.98)
s229d = s229.drop_redundant()

s280 = sample_of_density(0.95)
s280d = s280.drop_redundant()