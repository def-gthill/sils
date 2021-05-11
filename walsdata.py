import pandas as pd
import geopandas as gpd

langs = pd.read_csv('data/languages.csv')
langs_geo = gpd.GeoDataFrame(
    langs.copy(), geometry=gpd.points_from_xy(langs.Longitude, langs.Latitude)
)

features = pd.read_csv('data/parameters.csv')

values = pd.read_csv('data/values.csv')


