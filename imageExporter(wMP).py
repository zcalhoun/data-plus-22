# necessary imports

from pydoc import resolve
import ee
import pandas as pd
import numpy as np
import time
from geetools import batch
from tqdm import tqdm
from argparse import ArgumentParser
import requests
import math
import logging
import multiprocessing
import requests
import shutil
from retry import retry
import multiprocessing
import os
import csv
from functools import partial


def boundingBox(lat, lon, size, res):
    """ takes lat, lon of center point, desired size of image,
    and resolution of dataset to return coordinates of
    the four corners of the square centered at (lat, lon) of
    dimensions size

    :param lat: latitude of point of interest
    :type lat: float
    :param lon: longitude of point of interest
    :type lat: float
    :param size: size (in px) of desired image
    :type size: int
    :returns: coordinates (lat, lon) of bounding square corners
    :rtype: float
    """

    earth_radius = 6371000
    angular_distance = math.degrees(0.5 * ((size * res) / earth_radius))
    osLat = angular_distance
    osLon = angular_distance  # / math.cos(math.radians(lat))
    xMin = lon - osLon
    xMax = lon + osLon
    yMin = lat - osLat
    yMax = lat + osLat
    return xMin, xMax, yMin, yMax

# downloads image given URL


@retry(tries=10, delay=2, backoff=2)
def downloadImage(description, url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise response.raise_for_status()
        with open(os.path.join('/home/sl636/climateEye/exported_images', f'{description}.tif'), 'wb') as fd:
            fd.write(response.content)
        print(f'Done: {description}')
    except Exception as e:
        print(e)
        pass


dict = {'landsat': {'resolution': 30, 'RGB': ['B4', 'B3', 'B2'], 'NIR': 'B5', 'min': 0.0, 'max': 0.5},
        'naip': {'resolution': 1, 'RGB': ['R', 'G', 'B'], 'NIR': 'N', 'min': 0.0, 'max': 255.0},
        'sentinel': {'resolution': 10, 'RGB': ['B4', 'B3', 'B2'], 'NIR': 'B8', 'min': 0.0, 'max': 0.3}}

# generates URL to image centered around lat,lon and of desired size


@retry(tries=10, delay=2, backoff=2)
def generateURL(coord, height, width, res, filtered, crs, RGB):
    lat = coord[0]
    lon = coord[1]
    description = f"image_{lat}_{lon}"
    xMin, xMax, yMin, yMax = boundingBox(lat, lon, height, res)
    geometry = ee.Geometry.Rectangle([[xMin, yMin], [xMax, yMax]])
    filtered = filtered.filterBounds(geometry)
    try:
        image = filtered.median().clip(geometry)
        url = image.getDownloadUrl({
            'description': description,
            'region': geometry,
            'fileNamePrefix': description,
            'crs': crs,
            'fileFormat': 'GEO_TIFF',
            'bands': RGB,
            'region': geometry,
            'format': 'GEO_TIFF',
            'dimensions': [height, width]
        })
        urlPath = '/home/sl636/climateEye/URLs.txt'
        with open(urlPath, 'a') as fd:
            fd.write(f'\n{url}')

    except ee.EEException as e:
        # change below print to log
        print(e)
        print(f"\nimage at {(lat, lon)} missing one or more RGB bands")
        pass
    
    a = 'The image at ' +str(lat)+' '+ str(lon) + ' is exported.'
    logger.info(a)


if __name__ == "__main__":

    #  initialize GEE using project's service account and JSON key
    # service_account = 'climateeye@ee-saad-spike.iam.gserviceaccount.com'
    # credentials = ee.ServiceAccountCredentials(service_account, '/home/sl636/climateEye/ee-saad-spike-b059a4c480f6.json')
    # ee.Initialize(credentials)

    logger = logging.getLogger('my_logger')
    logging.basicConfig(filename='file.log',
                    filemode='a',
                    level = logging.INFO
                    )


    output_dir = '/home/sl636/climateEye/exported_images'
    try:
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    except:
        ee.Authenticate()
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        pass

# variables set for testing the script by simple run

    # path = '/home/sl636/climateEye/generated_coords_100.csv'
    # dataset = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
    # start = '2021-01-01'
    # end = '2022-06-01'
    # filtered = dataset.filterDate(start, end)

    parser = ArgumentParser()
    parser.add_argument("-f", "--filepath",
                        help="path to coordinates csv file",  type=str)
    parser.add_argument("-d", "--dataset", help="dataset to pull images from",
                        default=ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA"), type=ee.imagecollection.ImageCollection)
    parser.add_argument(
        "-r", "--resolution", help='resolution of passed dataset (in m/px)', default=30, type=int)
    parser.add_argument(
        "-s", "--start_date", help="start date for getting images", default='2017-01-01', type=str)
    parser.add_argument(
        "-e", "--end_date", help="end date for getting images", default='2018-01-01', type=str)
    parser.add_argument("-b", "--bands", help="list of desired bands",
                        default=['B2', 'B3', 'B4', 'B5'], type=list)
    parser.add_argument(
        "-he", "--height", help="height of output images (in px)", default=512, type=int)
    parser.add_argument(
        "-w", "--width", help="width of output images (in px)", default=512, type=int)
    parser.add_argument(
        "-o", "--output_dir", help="path to output directory", default="data/", type=str)

    args = parser.parse_args()

    # loops over all coords in csv, fetches URL of desired image, and appends it to URL.txt
    lat_lon_only = partial(generateURL, height=args.height, width=args.width,
                           res=dict['landsat']['resolution'], filtered=args.dataset.filterDate(args.start_date, args.end_date), crs='EPSG:3857', RGB=args.bands)

    with open(args.filepath, 'r') as coords_file:
        next(coords_file)
        coords = csv.reader(coords_file, quoting=csv.QUOTE_NONNUMERIC)
        data = list(coords)
    pool = multiprocessing.Pool()
    pool.map(lat_lon_only, data)
    pool.close()
    pool.join()

    # loops over all URLs in URL.txt and downloads images in parallel
    urlPath = '/home/sl636/climateEye/URLs.txt'
    with open(urlPath) as allURLs:
        pool2 = multiprocessing.Pool()
        export_start_time = time.time()
        pool2.starmap(downloadImage, enumerate(allURLs.read().split('\n')))
        export_finish_time = time.time()

    pool2.close()
    pool2.join()
    duration = export_finish_time - export_start_time
    DIR = output_dir
    num_downloaded = len([name for name in os.listdir(
        DIR) if os.path.isfile(os.path.join(DIR, name))])
    num_requested = len(pd.read_csv(args.filepath))
    print(
        f'Export complete! It took {duration:.2f} s ({duration/60:.2f} min) to download {num_downloaded} images out of {num_requested} requested')
