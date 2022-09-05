# data-plus-22
In this repository, the tool we wrote to download satellite images can be found under `imageExporter.py`. The tool requires a .csv file containing desired coordinates, the dataset to download images from (Sentinel, NAIP, Landsat), the desired dimensions of the output images, the output directory path, and whether
images should be pansharpened (only available for Landsat). 

Details on how to use `imageExporter.py' can be found below:

```
usage: imageExported.py [-h] [-f FILEPATH] [-d DATASET] [-s START_DATE] [-e END_DATE] [-he HEIGHT] [-w WIDTH] [-o OUTPUT_DIR] [-sh SHARPENED]

options:
  -h, --help            show this help message and exit
  -f FILEPATH, --filepath FILEPATH
                        path to coordinates csv file (assumptions: 2 columns only, top row: lon, lat)
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
                        
```

Since all arguments are set by default, the most basic way of running the code would be: 

```
python imageExporter.py -f FILEPATH
```

with FILEPATH replaced by the path to the desired coordinates csv file.

To run the tool, the user needs to have a personal or service account (we use a service account) to authenticate to Google Earth Engine, as well as a private JSON for that account. Instructions on how to create a service account can be found here: [Create Service Account](https://developers.google.com/earth-engine/guides/service_account#create-a-service-account). This link provides instructions to create the private JSON key: [Create JSON Key for Service Account](https://developers.google.com/earth-engine/guides/service_account#create-a-private-key-for-the-service-account).

Another requirement of the program is to install the Google Earth Engine. Instructions on how to do so can be found here: [Earth Engine Installation](https://developers.google.com/earth-engine/guides/python_install#install-options).

To install the rest of the required packages, the user can create a conda environment similar to the one we use. Our environment file can be found in `ee_env.yml`. Installing a conda environment using a yml file is done through: conda env create -f YML_FILE_NAME 
UPDATE: Some users reported issues using `ee_env.yml`. The issue was likely caused by build versions. We include another environment file `ee_env_no_builds.yml` to resolve that issue.

# Sampling Code
The sampling code can be found in the file, Sampling_from_Cities.ipynb, located in notebooks folder, main branch.

## Sampling Strategy
The 10,000,000 coordinates are created from the **10,000 most populated cities** and **multivariate normal distributions** centered at each city with standard deviation of **50 km** converted to degrees of latitude and longitude. The number of coordinates generated in each distribution depends on the proportion of _log(population)_. After over 10 M coordinates are generated (allowing room for coordinates not in land), we check the _overlap_ of those coordinates with the shape of the continents (except for Antarctica) from the **shape file** and only keeps the ones in land. If that ends up creating more than 10 M coordinates, we randomizes 10 M coordinates from that to keep and create a csv file that contains the coordinates generated.

## Dependencies
Three (3) files must be uploaded to run the code.
* **World_Continents.shp**: found in World_Continents folder
* **World_Continents.shx**: found in World_Continents folder
* **worldcities.csv**: found in data folder, document-sampler branch; contains all the cities of the world with information such as absolute location (lat, lon), population, etc.
