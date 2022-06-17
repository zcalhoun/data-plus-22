import numpy as np
import geopandas as gpd
import shapely.geometry

# source: https://gis.stackexchange.com/a/356502


def sample_geoseries(geoseries, size, overestimate=2):
    polygon = geoseries.unary_union
    min_x, min_y, max_x, max_y = polygon.bounds
    ratio = polygon.area / polygon.envelope.area
    samples = np.random.uniform(
        (min_x, min_y), (max_x, max_y), (int(size / ratio * overestimate), 2))
    multipoint = shapely.geometry.MultiPoint(samples)
    multipoint = multipoint.intersection(polygon)
    samples = np.array(multipoint)
    print(f'There were {samples.shape[0]} coordinates generated')
    a = samples[np.random.choice(len(samples), size)]
    np.savetxt(f"coords_{size}.csv", a, delimiter=",")


geodata = gpd.read_file(
    "/home/sl636/climateEye/World_Continents/World_Continents.shp")
points = sample_geoseries(geodata['geometry'], 100000)
