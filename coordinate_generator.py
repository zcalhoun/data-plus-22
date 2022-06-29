import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry

# source: https://gis.stackexchange.com/a/356502


def sample_geoseries(geoseries, size, filename, overestimate=2):
    polygon = geoseries.unary_union
    min_x, min_y, max_x, max_y = polygon.bounds
    ratio = polygon.area / polygon.envelope.area
    samples = np.random.uniform(
        (min_x, min_y), (max_x, max_y), (int(size / ratio * overestimate), 2))
    multipoint = shapely.geometry.MultiPoint(samples)
    multipoint = multipoint.intersection(polygon)
    samples = np.array(multipoint)
    a = samples[np.random.choice(len(samples), size)]
    header = np.array([['lon', 'lat']])
    file = np.vstack([header, a])
    pd.DataFrame(file).to_csv(
        f"/home/sl636/climateEye/{filename}_{size}.csv", header=None, index=None)


geodata = gpd.read_file(
    "/home/sl636/climateEye/World_Continents/World_Continents.shp")
no_antarctica = geodata.drop(6)
sample_geoseries(no_antarctica['geometry'], 10000, "no_antarctica")
# https://www.weather.gov/gis/USStates
us_data = gpd.read_file('/home/sl636/climateEye/s_22mr22/s_22mr22.shp')
sample_geoseries(us_data['geometry'], 10000, "america")
