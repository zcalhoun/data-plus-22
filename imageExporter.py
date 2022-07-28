from logging.config import valid_ident
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
    osLon = angular_distance
    xMin = lon - osLon
    xMax = lon + osLon
    yMin = lat - osLat
    yMax = lat + osLat
    return xMin, xMax, yMin, yMax


@retry(tries=10, delay=2, backoff=2)
def generateURL(coord, height, width, dataset, filtered, crs, output_dir, sharpened=False):
    """ generates the URL from Google Earth Engine of the image
    at coordinates coord, from filtered dataset and saves tif file
    to output_dir

    :param coord: longitude and latitude of desired image
    :type coord: tuple or list
    :param height: desired output image height 
    :type height: int
    :param width: desired output image width 
    :type width: int
    :param dataset: name of dataset (landsat, sentinel, naip)
    :type dataset: str
    :param filtered: filtered Earth Engine dataset (by date)
    :type filtered: ee.ImageCollection()
    :param crs: projection of the image (e.g. EPSG:3857)
    :type crs: str
    :param output_dir: path of output directory
    :type output_dir: str
    :param sharpened: whether we also download pansharpened images too
    :type sharpened: bool
    """

    lon = coord[0]
    lat = coord[1]
    description = f"{dataset}_image_{lat}_{lon}"
    res = dico[dataset]['resolution']
    xMin, xMax, yMin, yMax = boundingBox(lat, lon, height, res)
    geometry = ee.Geometry.Rectangle([[xMin, yMin], [xMax, yMax]])
    filtered = filtered.filterBounds(geometry)
    if dataset == 'sentinel':
        cloud_pct = 10
        filtered = filtered.filter(ee.Filter.lte(
            'CLOUDY_PIXEL_PERCENTAGE', cloud_pct))
    image = filtered.median().clip(geometry)
    RGB = dico[dataset]['RGB']
    _min = dico[dataset]['min']
    _max = dico[dataset]['max']
    band_names = image.bandNames()
    bands_list = band_names.getInfo()
    if all(band in bands_list for band in RGB):
        image_vis = image.visualize(bands=RGB, min=_min, max=_max)
        try:
            url = image_vis.getDownloadUrl({
                'description': description,
                'region': geometry,
                'fileNamePrefix': description,
                'crs': crs,
                'fileFormat': 'GEO_TIFF',
                'region': geometry,
                'format': 'GEO_TIFF',
                'dimensions': [height, width]
            })
            # download image given URL
            response = requests.get(url)
            if response.status_code != 200:
                raise response.raise_for_status()
            with open(os.path.join(output_dir, f'{description}.tif'), 'wb') as fd:
                fd.write(response.content)
            logging.info(f'Done: {description}')

        except Exception as e:
            logging.exception(e)

        panchromatic_band = dico[dataset]['panchromatic']
        if sharpened and panchromatic_band in bands_list:
            try:
                hsv = image.select(RGB).rgbToHsv()
                # Swap in the panchromatic band and convert back to RGB.
                # specific to Landsat as NAIP and Sentinel don't have a panchromatic band.
                sharpened = ee.Image.cat(
                    [hsv.select('hue'),
                     hsv.select('saturation'),
                     image.select(panchromatic_band)]).hsvToRgb()
            except Exception as e:
                logging.exception(e)
                pass

            try:
                sharpe_url = sharpened.getDownloadUrl({
                    'description': "sharpened"+description,
                    'region': geometry,
                    'fileNamePrefix': "sharpened"+description,
                    'crs': crs,
                    'fileFormat': 'GEO_TIFF',
                    'region': geometry,
                    'format': 'GEO_TIFF',
                    'dimensions': [height, width]
                })
                # download sharpened image given URL
                sharpe_response = requests.get(sharpe_url)
                if sharpe_response.status_code != 200:
                    raise sharpe_response.raise_for_status()
                with open(os.path.join(output_dir, f'sharpened_{description}.tif'), 'wb') as fd:
                    fd.write(sharpe_response.content)
                logging.info(f'Done: sharpened_{description}')

            except Exception as e:
                logging.exception(e)
                pass

    else:
        logging.info(f'Image at {(lat, lon)} has bands: {bands_list}')
        pass


if __name__ == "__main__":

    #  initialize GEE using project's service account and JSON key
    service_account = "climateeye@ee-saad-spike.iam.gserviceaccount.com"
    json_key = "/home/sl636/climateEye/ee-saad-spike-b059a4c480f6.json"
    ee.Initialize(
        ee.ServiceAccountCredentials(service_account, json_key), opt_url='https://earthengine-highvolume.googleapis.com')

    parser = ArgumentParser()
    parser.add_argument("-f", "--filepath",
                        help="path to coordinates csv file", default='/home/sl636/coordinates_generated_10000000-3.csv',  type=str)
    parser.add_argument("-d", "--dataset", help="name of dataset to pull images from (sentinel, landsat, or naip)",
                        default="sentinel", type=str)
    parser.add_argument(
        "-s", "--start_date", help="start date for getting images", default='2022-03-21', type=str)
    parser.add_argument(
        "-e", "--end_date", help="end date for getting images", default='2022-06-20', type=str)
    parser.add_argument(
        "-he", "--height", help="height of output images (in px)", default=512, type=int)
    parser.add_argument(
        "-w", "--width", help="width of output images (in px)", default=512, type=int)
    parser.add_argument(
        "-o", "--output_dir", help="path to output directory", default="output_images/", type=str)
    parser.add_argument(
        "-sh", "--sharpened", help="download pan-sharpened image (only available for Landsat)", default=False, type=bool)
    args = parser.parse_args()

    logging.basicConfig(
        filename=f'{args.dataset}_logger.log',
        filemode="w",
        level="INFO",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"Directory {args.output_dir} created")
    else:
        print("Please delete output directory before retrying")

    dico = {'landsat': {'dataset': ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA"), 'resolution': 30, 'RGB': ['B4', 'B3', 'B2'], 'NIR': 'B5', 'panchromatic': 'B8', 'min': 0.0, 'max': 0.4},
            'naip': {'dataset':  ee.ImageCollection("USDA/NAIP/DOQQ"), 'resolution': 1, 'RGB': ['R', 'G', 'B'], 'NIR': 'N', 'panchromatic': None, 'min': 0.0, 'max': 255.0},
            'sentinel': {'dataset': ee.ImageCollection("COPERNICUS/S2_SR"), 'resolution': 10, 'RGB': ['B4', 'B3', 'B2'], 'NIR': 'B8', 'panchromatic': None, 'min': 0.0, 'max': 4500.0}}

    lat_lon_only = partial(generateURL,
                           height=args.height,
                           width=args.width,
                           dataset=args.dataset,
                           filtered=dico[args.dataset]['dataset'].filterDate(
                               args.start_date, args.end_date),
                           crs='EPSG:3857',
                           output_dir=args.output_dir,
                           sharpened=args.sharpened)

    with open(args.filepath, 'r') as coords_file:
        next(coords_file)
        coords = csv.reader(coords_file, quoting=csv.QUOTE_NONNUMERIC)
        data = list(coords)

    DIR = args.output_dir
    num_downloaded = len([name for name in os.listdir(
        DIR) if os.path.isfile(os.path.join(DIR, name))])

    for i in range(0, len(data), 10000):
        pool = multiprocessing.Pool()
        export_start_time = time.time()
        print(f"Starting rows: {i} to {i+10000}")
        logging.info(f"Starting rows: {i} to {i+10000}")
        pool.map(lat_lon_only, data[i:i+10000])
        export_finish_time = time.time()
        pool.close()
        pool.join()
        logging.info(f"Finished rows: {i} to {i+10000}")
        logging.info(f"Downloaded {num_downloaded} images so far")
        print(f"Finished rows: {i} to {i+10000}")

    duration = export_finish_time - export_start_time
    num_requested = len(pd.read_csv(args.filepath))
    result = f'Export complete! It took {duration:.2f} s ({duration/60:.2f} min) to download {num_downloaded} images out of {2*num_requested if args.sharpened else num_requested} requested from {args.filepath} using the {args.dataset} dataset'
    logging.info(result)
    print(result)
    with open('results.txt', 'a') as f:
        f.write(result+'\n')
