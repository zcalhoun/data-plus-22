# data-plus-22
In this repository, the tool we wrote to download satellite images can be found under `imageExporter.py`. The tool requires a .csv file containing desired coordinates, the dataset to download images from (Sentinel, NAIP, Landsat), the desired dimensions of the output images, the output directory path, and whether
images should be pansharpened (only available for Landsat).

Details on how to use `imageExporter.py' can be found below:

usage: imageExported.py [-h] [-f FILEPATH] [-d DATASET] [-s START_DATE] [-e END_DATE] [-he HEIGHT] [-w WIDTH] [-o OUTPUT_DIR] [-sh SHARPENED]

options:
  -h, --help            show this help message and exit
  -f FILEPATH, --filepath FILEPATH
                        path to coordinates csv file
  -d DATASET, --dataset DATASET
                        name of dataset to pull images from (sentinel, landsat, or naip)
  -s START_DATE, --start_date START_DATE
                        start date for getting images
  -e END_DATE, --end_date END_DATE
                        end date for getting images
  -he HEIGHT, --height HEIGHT
                        height of output images (in px)
  -w WIDTH, --width WIDTH
                        width of output images (in px)
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        path to output directory
  -sh SHARPENED, --sharpened SHARPENED
                        download pan-sharpened image (only available for Landsat)

To run the tool, the user needs to have a personal or service account (we use a service account) to authenticate to Google Earth Engine, as well as a private JSON for that account. Instructions on how to create a service account can be found here: [Create Service Account](https://developers.google.com/earth-engine/guides/service_account#create-a-service-account). This link provides instructions to create the private JSON key: [Create JSON Key for Service Account](https://developers.google.com/earth-engine/guides/service_account#create-a-private-key-for-the-service-account).

Another requirement of the program is to install the Google Earth Engine. Instructions on how to do so can be found here: [Earth Engine Installation](https://developers.google.com/earth-engine/guides/python_install#install-options).

To install the rest of the required packages, the user can create a conda environment similar to the one we use. Our environment file can be found in `ee_env.yml`.

