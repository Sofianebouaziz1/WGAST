import ee
import pandas as pd

class Sentinel2Processor:
    def __init__(self, start_date, end_date, bounds):
        """
        Initialize the Sentinel2Processor with study area and date range.
        
        Parameters:
        - start_date (str): The start date of the image collection (format: 'YYYY-MM-DD').
        - end_date (str): The end date of the image collection (format: 'YYYY-MM-DD').
        - bounds (list): List of coordinates defining the area of interest (AOI) in [xmin, ymin, xmax, ymax] format.
        """
        # Initialize the Earth Engine module
        # ee.Authenticate()  # Uncomment if authentication is needed
        ee.Initialize()

        # Define the study area and date range
        self.aoi = ee.Geometry.Rectangle(bounds)
        self.start_date = start_date
        self.end_date = end_date


    def get_Sentinel2_collection(self):
        """
        Get the Sentinel-2 image collection, apply cloud masking, and filter by pixel availability.
        """
        return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate(self.start_date, self.end_date) \
            .filterBounds(self.aoi) \
            .map(self.applyScaleFactors) \
            .map(self.maskS2clouds) \
            .map(self.calculatePixelAvailability) \
            .map(self.calculate_indices)
    
    def applyScaleFactors(self, image):
        """
        Apply scale factors to Blue, Green, and Red bands.
        """
        # Select bands 2, 3, and 4 (Blue, Green, Red)
        scaled_image = image.select(['B2', 'B3', 'B4', 'B8', 'B11']).multiply(0.0001)
        return image.addBands(scaled_image, None, True)


    def maskS2clouds(self, image):
        """
        Apply a cloud mask to the Sentinel-2 image using the SCL band.
        """
        # Selecting Cloudy Mask
        cloud_shadow = image.select('SCL').eq(3)
        cloud_low = image.select('SCL').eq(7)
        cloud_med = image.select('SCL').eq(8)
        cloud_high = image.select('SCL').eq(9)
        
        cloud_mask = cloud_shadow.add(cloud_low).add(cloud_med).add(cloud_high)
        
        # Inverting the Mask
        mask = cloud_mask.eq(0).selfMask()
        
        return image.updateMask(mask)
    
    def calculatePixelAvailability(self, image):
        """
        Calculate the percentage of valid pixels (non-masked) in the image.
        """
        total_pixels = image.select('SCL').mask().reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=self.aoi,
            scale=60,
            maxPixels=1e9
        ).values().get(0)

        valid_pixels = image.select('SCL').reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=self.aoi,
            scale=60,
            maxPixels=1e9
        ).values().get(0)

        pixel_availability = ee.Number(valid_pixels).divide(total_pixels).multiply(100)
        return image.set('pixelAvailability', pixel_availability)
    

    def filter_disponible_images(self, collection, percentage):
        """
        Filter the image collection based on the percentage of valid (non-masked) pixels.
        
        Parameters:
        - collection (ee.ImageCollection): The image collection to filter.
        - percentage (float): The minimum percentage of valid pixels required.
        
        Returns:
        - ee.ImageCollection: The filtered image collection.
        """
        collection = collection.filter(ee.Filter.gte('pixelAvailability', percentage))
        return collection#.map(self.applyScaleFactors)

    def countImages(self, collection):
        """
        Count the number of images in the filtered collection.
        """
        return collection.size().getInfo()
     
    def calculate_indices(self, image):
        """
        Calculate the spectral indices.
        """   
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
        ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")
        return image.addBands([ndvi, ndwi, ndbi])


    def get_times(self, collection):
        """
        Get the times of the images in the collection.
        """
        dates = collection.aggregate_array('system:time_start') \
            .map(lambda ele: ee.Date(ele).format())

        return dates.getInfo()
    

    def get_SR(self, collection):
        return collection.select(['B2', 'B3', 'B4'])
    

    def get_index(self, collection):
        return collection.select(['NDVI', 'NDWI', 'NDBI'])
    
