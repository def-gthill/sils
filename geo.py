"""
Geographical groups based on a DBSCAN clustering of the s280d sample
"""

import math

from sklearn import cluster

from walsdata import s280d


dbscan = cluster.DBSCAN(metric='haversine', eps=0.12, min_samples=3)
labels = dbscan.fit_predict(s280d.langs[['Latitude', 'Longitude']].applymap(math.radians))

region_names = {
    -1: 'Outlier',
    0: 'Paraguay',
    1: 'Caucasus',
    2: 'USA/Canada',
    3: 'Africa',
    4: 'New Guinea',
    5: 'Europe',
    6: 'Andes/Amazon',
    7: 'Southeast Asia',
    8: 'Taiwan',
    9: 'Pakistan',
    10: 'Mexico',
    11: 'West Russia',
    12: 'Melanesia',
    13: 'Australia',
    14: 'Indonesia/Malaysia',
}

region_labels = [region_names[label] for label in labels]
