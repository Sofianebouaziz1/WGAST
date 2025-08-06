import fnmatch
import os
import rasterio
import pandas as pd
import numpy as np

class GetTriple:

    def read_file(self, path):
        """ load csv file """
        if os.path.exists(path):
            with rasterio.open(path) as src:
                return src.read(1)

    def create_mask(self, image):
        """ Create a mask where 0.0 in the image corresponds to 0 in the mask, and everything else is 1"""
        mask = np.where(image == 0.0, 0, 1)
        return mask
    
    def load_sentinel(self, path, dates):
        """Load Sentinel images (first three bands) along with their CRS and transform."""
        sentinel_images = []

        # Format the dates to match the Sentinel file naming convention (e.g., '20170409')
        common_dates = pd.Series(dates)
        formatted_dates = common_dates.apply(lambda x: pd.to_datetime(x).strftime('%Y%m%d')).tolist()

        for date in formatted_dates:
            # Construct the expected filename pattern
            filename_pattern = f"{date}*_T31UDP.tif"
            matching_file = [f for f in os.listdir(path) if fnmatch.fnmatch(f, filename_pattern)][0]
            file_path = os.path.join(path, matching_file)

            if os.path.exists(file_path):
                print(f"Reading Sentinel file: {file_path}")
                with rasterio.open(file_path) as src:
                    # Read the first three bands
                    bands = src.read([1, 2, 3])  # Read bands 1, 2, and 3
                    crs = src.crs  # Get CRS
                    transform = src.transform  # Get transform (affine matrix)
                    
                    sentinel_images.append((bands, crs, transform))  # Append as a tuple (bands, crs, transform)

        return sentinel_images


    def load_landsat(self, path, dates):
        """Load Landsat images along with their CRS and transform."""
        landsat_images = []

        common_dates = pd.Series(dates)
        formatted_dates = common_dates.apply(lambda x: pd.to_datetime(x).strftime('%Y%m%d')).tolist()

        for date in formatted_dates:
            filename_pattern = f"LC08_199027_{date}.tif"
            file_path = os.path.join(path, filename_pattern)
              
            if os.path.exists(file_path):
                print(f"Reading Landsat file: {file_path}")
                with rasterio.open(file_path) as src:
                    array = src.read([1, 2, 3, 4])  # Read the first band (image data)
                    crs = src.crs  # Get CRS
                    transform = src.transform  # Get transform (affine matrix)
                    
                    landsat_images.append((array, crs, transform))  # Append as a tuple (image, crs, transform)

        return landsat_images

    def load_modis(self, path, dates):
        """Load MODIS images along with their CRS and transform."""
        modis_images = []

        common_dates = pd.Series(dates)
        formatted_dates = common_dates.apply(lambda x: pd.to_datetime(x).strftime('%Y_%m_%d')).tolist()

        for date in formatted_dates:
            filename_pattern = f"{date}.tif"  # Assuming MODIS files are saved with .tif extension
            file_path = os.path.join(path, filename_pattern)

            if os.path.exists(file_path):
                print(f"Reading MODIS file: {file_path}")
                with rasterio.open(file_path) as src:
                    array = src.read(1)  # Read the first band (image data)
                    crs = src.crs  # Get CRS
                    transform = src.transform  # Get transform (affine matrix)
                    
                    modis_images.append((array, crs, transform))  # Append as a tuple (image, crs, transform)

        return modis_images
    

    def save_sentinel_formatted(self, sentinel_images, dates, output_folder):
        """Save formatted Landsat images to the specified output folder."""
        dates = pd.Series(dates)
        
        for (image, crs, transform), date in zip(sentinel_images, dates):

            for band in range(image.shape[0]):  # Iterate over each band
                image[band][np.isnan(image[band])] = 0
        
            filename = f"S_{pd.to_datetime(date).strftime('%Y%m%d')}.tif"
            file_path = os.path.join(output_folder, filename)

            mask_image = self.create_mask(image[0])
            filename_mask = f"S_mask_{pd.to_datetime(date).strftime('%Y%m%d')}"
            file_path_mask= os.path.join(output_folder, filename_mask)

            np.save(file_path_mask, mask_image)

            with rasterio.open(
                file_path,
                'w',
                driver='GTiff',
                height=image[0].shape[0],
                width=image[0].shape[1],
                count=3,  # Assuming 1 band
                dtype=image.dtype,
                crs=crs,  # Use the CRS from the input file
                transform=transform  # Use the transform from the input file
            ) as dst:
                for band_idx in range(3):  # Write each band
                    dst.write(image[band_idx], band_idx + 1)

    def save_sentinel_augmented_formatted(self, sentinel_images, dates, output_folder):
        """Save formatted Landsat images to the specified output folder."""
        dates = pd.Series(dates)
        
        # Loop through each date and corresponding images
        for images_list, date in zip(sentinel_images, dates):
            # Iterate over the four rotated images in the sublist
            for idx, (rotated_image, crs, transform) in enumerate(images_list):  # Each images_list contains 4 images (rotated)
                for band in range(rotated_image.shape[0]):  # Iterate over each band
                    rotated_image[band][np.isnan(rotated_image[band])] = 0
        

                # Generate the filename with the corresponding angle and date
                filename = f"S_{idx}_{pd.to_datetime(date).strftime('%Y%m%d')}.tif"
                file_path = os.path.join(output_folder, filename)

                with rasterio.open(
                        file_path,
                        'w',
                        driver='GTiff',
                        height=rotated_image[0].shape[0],
                        width=rotated_image[0].shape[1],
                        count=3,  # Assuming 1 band
                        dtype=rotated_image.dtype,
                        crs=crs,  # Use the CRS from the input file
                        transform=transform  # Use the transform from the input file
                    ) as dst:
                        for band_idx in range(3):  # Write each band
                            dst.write(rotated_image[band_idx], band_idx + 1)

                # Create and save the mask for the rotated image
                mask_image = self.create_mask(rotated_image[0])  # Assuming you have a create_mask function
                filename_mask = f"S_mask_{idx}_{pd.to_datetime(date).strftime('%Y%m%d')}"
                file_path_mask = os.path.join(output_folder, filename_mask)
                np.save(file_path_mask, mask_image)

    

    def save_sentinel_formatted(self, sentinel_images, dates, output_folder):
        """Save formatted Landsat images to the specified output folder."""
        dates = pd.Series(dates)
        
        for (image, crs, transform), date in zip(sentinel_images, dates):

            for band in range(image.shape[0]):  # Iterate over each band
                image[band][np.isnan(image[band])] = 0
        
            filename = f"S_{pd.to_datetime(date).strftime('%Y%m%d')}.tif"
            file_path = os.path.join(output_folder, filename)

            mask_image = self.create_mask(image[0])
            filename_mask = f"S_mask_{pd.to_datetime(date).strftime('%Y%m%d')}"
            file_path_mask= os.path.join(output_folder, filename_mask)

            np.save(file_path_mask, mask_image)

            with rasterio.open(
                file_path,
                'w',
                driver='GTiff',
                height=image[0].shape[0],
                width=image[0].shape[1],
                count=3,  # Assuming 1 band
                dtype=image.dtype,
                crs=crs,  # Use the CRS from the input file
                transform=transform  # Use the transform from the input file
            ) as dst:
                for band_idx in range(3):  # Write each band
                    dst.write(image[band_idx], band_idx + 1)

    def save_landsat_formatted(self, landsat_images, dates, output_folder):
        """Save formatted Landsat images to the specified output folder."""
        dates = pd.Series(dates)
        
        for (image, crs, transform), date in zip(landsat_images, dates):
            # Replace NaN values with 0
            for band in range(image.shape[0]):  # Iterate over each band
                image[band][np.isnan(image[band])] = 0

            filename = f"L_{pd.to_datetime(date).strftime('%Y%m%d')}.tif"
            file_path = os.path.join(output_folder, filename)


            mask_image = self.create_mask(image[0])
            filename_mask = f"L_mask_{pd.to_datetime(date).strftime('%Y%m%d')}"
            file_path_mask= os.path.join(output_folder, filename_mask)

            np.save(file_path_mask, mask_image)

            with rasterio.open(
                file_path,
                'w',
                driver='GTiff',
                height=image[0].shape[0],
                width=image[0].shape[1],
                count=4,  # Assuming 1 band
                dtype=image.dtype,
                crs=crs,  # Use the CRS from the input file
                transform=transform  # Use the transform from the input file
            ) as dst:
                for band_idx in range(4):  # Write each band
                    dst.write(image[band_idx], band_idx + 1)


    def save_landsat_augmented_formatted(self, landsat_images, dates, output_folder):
        """Save formatted Landsat images to the specified output folder."""
        dates = pd.Series(dates)
        
        # Loop through each date and corresponding images
        for images_list, date in zip(landsat_images, dates):
            # Iterate over the four rotated images in the sublist
            for idx, (rotated_image, crs, transform) in enumerate(images_list):  # Each images_list contains 4 images (rotated)
                # Generate the filename with the corresponding angle and date
                filename = f"L_{idx}_{pd.to_datetime(date).strftime('%Y%m%d')}.tif"
                file_path = os.path.join(output_folder, filename)

                # Save the rotated image
                with rasterio.open(
                    file_path,
                    'w',
                    driver='GTiff',
                    height=rotated_image.shape[0],
                    width=rotated_image.shape[1],
                    count=1,  # Assuming 1 band
                    dtype=rotated_image.dtype,
                    crs=crs,  # Use the CRS from the input file
                    transform=transform  # Use the transform from the input file
                ) as dst:
                    dst.write(rotated_image, 1)

                # Create and save the mask for the rotated image
                mask_image = self.create_mask(rotated_image)  # Assuming you have a create_mask function
                filename_mask = f"L_mask_{idx}_{pd.to_datetime(date).strftime('%Y%m%d')}"
                file_path_mask = os.path.join(output_folder, filename_mask)
                np.save(file_path_mask, mask_image)


          
    def save_modis_formatted(self, modis_images, dates, output_folder):
        """Save formatted MODIS images to the specified output folder."""
        dates = pd.Series(dates)

        for (image, crs, transform), date in zip(modis_images, dates):
            # Replace NaN values with 0
            image[np.isnan(image)] = 0  # Replace NaNs with 0

            filename = f"M_{pd.to_datetime(date).strftime('%Y%m%d')}.tif"
            file_path = os.path.join(output_folder, filename)
            
            mask_image = self.create_mask(image)
            filename_mask = f"M_mask_{pd.to_datetime(date).strftime('%Y%m%d')}.npy"
            file_path_mask = os.path.join(output_folder, filename_mask)

            np.save(file_path_mask, mask_image)

            with rasterio.open(
                file_path,
                'w',
                driver='GTiff',
                height=image.shape[0],
                width=image.shape[1],
                count=1,  # Assuming 1 band
                dtype=image.dtype,
                crs=crs,  # Use the CRS from the input file
                transform=transform  # Use the transform from the input file
            ) as dst:
                dst.write(image, 1)


    def save_modis_augmented_formatted(self, modis_images, dates, output_folder):
        """Save formatted Landsat images to the specified output folder."""
        dates = pd.Series(dates)
        
        # Loop through each date and corresponding images
        for images_list, date in zip(modis_images, dates):
            # Iterate over the four rotated images in the sublist
            for idx, (rotated_image, crs, transform) in enumerate(images_list):  # Each images_list contains 4 images (rotated)
                # Generate the filename with the corresponding angle and date
                filename = f"M_{idx}_{pd.to_datetime(date).strftime('%Y%m%d')}.tif"
                file_path = os.path.join(output_folder, filename)

                # Save the rotated image
                with rasterio.open(
                    file_path,
                    'w',
                    driver='GTiff',
                    height=rotated_image.shape[0],
                    width=rotated_image.shape[1],
                    count=1,  # Assuming 1 band
                    dtype=rotated_image.dtype,
                    crs=crs,  # Use the CRS from the input file
                    transform=transform  # Use the transform from the input file
                ) as dst:
                    dst.write(rotated_image, 1)

                # Create and save the mask for the rotated image
                mask_image = self.create_mask(rotated_image)  # Assuming you have a create_mask function
                filename_mask = f"M_mask_{idx}_{pd.to_datetime(date).strftime('%Y%m%d')}"
                file_path_mask = os.path.join(output_folder, filename_mask)
                np.save(file_path_mask, mask_image)
