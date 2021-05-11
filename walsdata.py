import pandas as pd
import geopandas as gpd

langs = pd.read_csv('data/languages.csv')
langs_geo = gpd.GeoDataFrame(
    langs.copy(), geometry=gpd.points_from_xy(langs.Longitude, langs.Latitude)
)

features = pd.read_csv('data/parameters.csv')

values = pd.read_csv('data/values.csv')
present_values = pd.crosstab(values.Language_ID, values.Parameter_ID)


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
    def __init__(self, density_threshold):
        self.present_values = choose_and_evaluate_features_and_languages(
            present_values,
            density_threshold=density_threshold,
            n_features_to_drop=1,
            n_languages_to_drop=2,
            verbose=False,
        )
        self.langs_list = list(self.present_values.index)
        self.langs = langs_geo[langs_geo.ID.isin(self.langs_list)]
        self.lang_names = list(self.langs.Name)
        self.features_list = list(self.present_values.columns)
        self.features = features[features.ID.isin(self.features_list)]
        self.feature_names = list(self.features.Name)
        self.values = values[
            values.Language_ID.isin(self.langs.ID) &
            values.Parameter_ID.isin(self.features.ID)
        ]


s229 = Sample(0.98)
s280 = Sample(0.95)
