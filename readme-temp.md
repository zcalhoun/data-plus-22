# Sampling Code
The sampling code can be found in the file, Sampling_from_Cities.ipynb, located in notebooks folder, main branch.

## Sampling Strategy
The 10,000,000 coordinates are created from the **10,000 most populated cities** and **multivariate normal distributions** centered at each city with standard deviation of **50 km** converted to degrees of latitude and longitude. The number of coordinates generated in each distribution depends on the proportion of _log(population)_. After over 10 M coordinates are generated (allowing room for coordinates not in land), we check the _overlap_ of those coordinates with the shape of the continents (except for Antarctica) from the **shape file** and only keeps the ones in land. If that ends up creating more than 10 M coordinates, we randomizes 10 M coordinates from that to keep and create a csv file that contains the coordinates generated.

## Dependencies
Three (3) files must be uploaded to run the code.
* **World_Continents.shp**: found in World_Continents folder
* **World_Continents.shx**: found in World_Continents folder
* **worldcities.csv**: found in data folder, document-sampler branch; contains all the cities of the world with information such as absolute location (lat, lon), population, etc.
