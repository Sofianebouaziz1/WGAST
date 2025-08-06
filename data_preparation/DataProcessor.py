import numpy as np
from scipy.ndimage import generic_filter
import torch
import torch.nn.functional as F


class DataProcessor:
    """
    A class to process different types of images, including applying a progressive focal mean.
    The images should be provided as numpy arrays with associated CRS and transform data.
    """
    
    def progressive_focal_mean(self, images_with_metadata, initial_size=3):
        """
        Apply progressive focal mean to a numpy array of images, preserving CRS and transform information.
        
        Args:
            images_with_metadata (list of tuples): Each element is a tuple (image, crs, transform), 
                                                   where `image` is a numpy array, 
                                                   `crs` is the coordinate reference system, 
                                                   and `transform` is the affine transform.
            initial_size (int): Initial window size for focal mean (default is 3).
        
        Returns:
            List of tuples: Processed images with the same structure (image, crs, transform).
        """
        processed_images = []

        count = 1
        
        for (image, crs, transform) in images_with_metadata:
            # Create a copy of the image to work on
            image_cleaned = image.copy()

            # Replace cloud-covered pixels (value 0) with NaN
            image_cleaned = np.where(image_cleaned == 0, np.nan, image_cleaned)

            # Apply progressive focal mean
            self._apply_progressive_focal_mean(image_cleaned, initial_size)

            # Append the processed image with the original CRS and transform
            processed_images.append((image_cleaned, crs, transform))
            
            print("Image number : ", count, " has been filled")
            count = count + 1

        return processed_images


    def progressive_focal_mean_multiband(self, images_with_metadata, initial_size=3, step_size = 2):
        """
        Apply progressive focal mean to a multi-band numpy array of images, preserving CRS and transform information.
        
        Args:
            images_with_metadata (list of tuples): Each element is a tuple (image, crs, transform), 
                                                   where `image` is a numpy array with multiple bands,
                                                   `crs` is the coordinate reference system, 
                                                   and `transform` is the affine transform.
            initial_size (int): Initial window size for focal mean (default is 3).
        
        Returns:
            List of tuples: Processed images with the same structure (image, crs, transform).
        """
        processed_images = []

        count = 1
        
        for (image, crs, transform) in images_with_metadata:
            # Create a copy of the image to work on
            image_cleaned = image.copy()
            
            # Process each band separately
            for band in range(image_cleaned.shape[0]):
                # Replace cloud-covered pixels (value 0) with NaN
                image_cleaned[band] = np.where(image_cleaned[band] == 0, np.nan, image_cleaned[band])
                
                # Apply progressive focal mean
                self._apply_progressive_focal_mean(image_cleaned[band], initial_size, step_size)

            # Append the processed image with the original CRS and transform
            processed_images.append((image_cleaned, crs, transform))
            
            print("Image number:", count, "has been filled")
            count += 1

        return processed_images
    
    
    def _apply_progressive_focal_mean(self, image, initial_size, step_size = 2):
        """
        Apply progressive focal mean to a single image, replacing NaN values iteratively.
        
        Args:
            image (numpy array): The image to process.
            initial_size (int): Initial window size for the mean filter.
        """
        current_size = initial_size
        while np.isnan(image).any():
            # Apply a focal mean to the NaN pixels using the current window size
            focal_mean_image = generic_filter(image, self._local_mean, size=current_size, mode='constant', cval=np.nan)

            # Update only the NaN pixels with the focal mean values
            nan_mask = np.isnan(image)
            image[nan_mask] = focal_mean_image[nan_mask]

            # Increase the window size for the next iteration
            current_size += step_size  # Increase by 2 to move to the next odd size (e.g., 5, 7, 9,...)

    def _local_mean(self, arr):
        """
        Calculate the local mean of non-NaN values for a window in the image.
        
        Args:
            arr (numpy array): A window of pixel values.
        
        Returns:
            float: Mean of the non-NaN values in the window.
        """
        valid_vals = arr[~np.isnan(arr)]
        if len(valid_vals) == 0:
            return np.nan  # If all neighbors are NaN, return NaN
        return np.mean(valid_vals)


    def resize_modis_images(self, modis_images_with_metadata, target_height, target_width):
        """Resize MODIS images to the specified target dimensions using cubic interpolation."""
        resized_images_with_metadata = []

        for image, crs, transform in modis_images_with_metadata:

            image = np.where(image == 0, np.nan, image)

            # Convert the image to a PyTorch tensor and add required dimensions
            modis_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

            # Resize the image using bicubic interpolation
            modis_resized_bicubic = F.interpolate(modis_tensor, size=(target_height, target_width), mode='bicubic', align_corners=True)

            # Convert back to a NumPy array
            resized_image = modis_resized_bicubic.squeeze().numpy()

            # Append the resized image along with its CRS and transform
            resized_images_with_metadata.append((resized_image, crs, transform))

        return resized_images_with_metadata
    


    def resize_landsat_images(self, landsat_images_with_metadata, target_height, target_width):
        """Resize Landsat images to the specified target dimensions using cubic interpolation for each band."""
        resized_images_with_metadata = []

        for image, crs, transform in landsat_images_with_metadata:
            
            # Initialize an empty list to hold resized bands
            resized_bands = []

            # Iterate over each band in the image
            for band in range(image.shape[0]):  # Assuming the shape is (num_bands, height, width)
                band_image = image[band]  # Extract the band
                
                # ADDED: Convert 0 values to NaN (if needed)
                band_image = np.where(band_image == 0, np.nan, band_image)

                # Convert the band image to a PyTorch tensor and add required dimensions
                band_tensor = torch.from_numpy(band_image).unsqueeze(0).unsqueeze(0)

                # Resize the band using bicubic interpolation
                resized_band_bicubic = F.interpolate(band_tensor, size=(target_height, target_width), mode='bicubic', align_corners=True)

                # Convert back to a NumPy array
                resized_band = resized_band_bicubic.squeeze().numpy()

                # Append the resized band to the list
                resized_bands.append(resized_band)

            # Stack the resized bands along the first axis (to form a multi-band image)
            resized_image = np.stack(resized_bands)

            # Append the resized image along with its CRS and transform
            resized_images_with_metadata.append((resized_image, crs, transform))

        return resized_images_with_metadata




    def resize_modis_images2(self, modis_images_with_metadata, target_height, target_width):
        """Resize MODIS images to the specified target dimensions using cubic interpolation."""
        resized_images_with_metadata = []

        for image in modis_images_with_metadata:
            # Convert the image to a PyTorch tensor and add required dimensions
            modis_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

            # Resize the image using bicubic interpolation
            modis_resized_bicubic = F.interpolate(modis_tensor, size=(target_height, target_width), mode='bicubic', align_corners=True)

            # Convert back to a NumPy array
            resized_image = modis_resized_bicubic.squeeze().numpy()

            # Append the resized image along with its CRS and transform
            resized_images_with_metadata.append((resized_image))

        return resized_images_with_metadata