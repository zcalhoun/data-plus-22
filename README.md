# data-plus-22
In this repository, the tool we wrote to download satellite images can be found under imageExporter.py. The tool requires a .csv file containing desired coordinates, the dataset to download images from (Sentinel, NAIP, Landsat), the desired dimensions of the output images, the output directory path, and whether
images should be pansharpened (only available for Landsat).

The file 10MCoordinates.csv contains the 10 million coordinates we generated from a population-weighted sampling strategy using the 10k most populated cities. 