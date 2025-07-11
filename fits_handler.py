#!/usr/bin/env python3
"""
FITS Image Handler
Separate class to handle FITS file loading and management for the AllSkyCam dashboard.
"""

import os
import glob
import logging

logger = logging.getLogger(__name__)

class FitsImageHandler:
    """Handle FITS file loading and management."""
    
    def __init__(self, images_directory="images"):
        """Initialize FITS handler.
        
        Args:
            images_directory: Directory containing FITS files
        """
        self.images_directory = images_directory
        self.fits_files = []
        self.current_index = 0
        self.load_fits_files()
        
    def load_fits_files(self):
        """Load all FITS files from the images directory."""
        # Search for FITS files in images directory and subdirectories
        fits_patterns = [
            f"{self.images_directory}/*.fits",
            f"{self.images_directory}/**/*.fits"
        ]
        
        self.fits_files = []
        for pattern in fits_patterns:
            self.fits_files.extend(glob.glob(pattern, recursive=True))
        
        # Sort files by name for consistent ordering
        self.fits_files.sort()
        
        if self.fits_files:
            logger.info(f"Loaded {len(self.fits_files)} FITS files")
        else:
            logger.warning(f"No FITS files found in {self.images_directory}")
            
    def get_file_count(self):
        """Get total number of FITS files."""
        return len(self.fits_files)
        
    def get_current_index(self):
        """Get current file index."""
        return self.current_index
        
    def get_current_filename(self):
        """Get current filename."""
        if self.fits_files and 0 <= self.current_index < len(self.fits_files):
            return os.path.basename(self.fits_files[self.current_index])
        return "No file"
        
    def load_current_image(self):
        """Load the current FITS image as numpy array.
        
        Returns:
            tuple: (image_data as numpy array, filename)
        """
        if not self.fits_files or self.current_index < 0 or self.current_index >= len(self.fits_files):
            return None, "No files available"
            
        try:
            file_path = self.fits_files[self.current_index]
            
            # Handle FITS files
            try:
                from astropy.io import fits
                with fits.open(file_path) as hdul:
                    image_data = hdul[0].data
                    filename = os.path.basename(file_path)
                    logger.info(f"Loaded FITS image: {filename}")
                    return image_data, filename
            except ImportError:
                logger.error("astropy not available for FITS support")
                return None, "astropy required for FITS"
                
        except Exception as e:
            logger.error(f"Failed to load FITS image: {e}")
            return None, f"Error: {e}"
            
    def next_image(self):
        """Move to next image."""
        if self.fits_files and len(self.fits_files) > 0:
            self.current_index = (self.current_index + 1) % len(self.fits_files)
            return self.load_current_image()
        return None, "No files"
        
    def previous_image(self):
        """Move to previous image."""
        if self.fits_files and len(self.fits_files) > 0:
            self.current_index = (self.current_index - 1) % len(self.fits_files)
            return self.load_current_image()
        return None, "No files"
        
    def goto_image(self, index):
        """Go to specific image by index.
        
        Args:
            index: Index of the image to load
            
        Returns:
            tuple: (image_data, filename)
        """
        if 0 <= index < len(self.fits_files):
            self.current_index = index
            return self.load_current_image()
        return None, "Invalid index"
