"""
Unified CameraAcquisition Class - Combines features from both implementations
"""
import logging
import os
import threading
import time
import inspect
from datetime import datetime, UTC
from typing import Optional, Tuple, Any
import numpy as np
from astropy.io import fits

try:
    from ids_peak import ids_peak
    from ids_peak import ids_peak_ipl_extension
    from ids_peak_ipl import ids_peak_ipl
    HAS_IDS_PEAK = True
except ImportError:
    logging.error("Failed to import IDS Peak libraries. Please ensure they are installed correctly.")
    HAS_IDS_PEAK = False

logger = logging.getLogger(__name__)


class CameraAcquisition:
    """
    Unified camera acquisition class that combines simple batch processing
    and advanced continuous acquisition capabilities.
    
    Features:
    - Simple batch acquisition mode (like simpleExposure)
    - Advanced single-image acquisition mode (like allskyController)
    - Thread-safe operations
    - Comprehensive error handling
    - Multiple output formats (FITS, JPEG)
    - GPS metadata integration
    - Hot pixel correction
    """
    
    def __init__(self, 
                 exposure_time_ms: float = 10.0,
                 num_buffers: Optional[int] = None,
                 output_dir: str = "output",
                 hotpixel_correction: bool = False,
                 enable_jpeg_output: bool = False,
                 enable_auto_gain: bool = False):
        """
        Initialize the camera acquisition system.
        
        Args:
            exposure_time_ms: Exposure time in milliseconds
            num_buffers: Number of buffers to allocate (None = use minimum required)
            output_dir: Directory to save output files
            hotpixel_correction: Enable hot pixel correction
            enable_jpeg_output: Enable JPEG output with overlays
            enable_auto_gain: Enable automatic gain control
        """
        # Basic configuration
        self.exposure_time_ms = exposure_time_ms
        self.num_buffers = num_buffers
        self.output_dir = output_dir
        self.hotpixel_correction = hotpixel_correction
        self.enable_jpeg_output = enable_jpeg_output
        self.enable_auto_gain = enable_auto_gain
        
        # Hardware objects
        self.device = None
        self.data_stream = None
        self.remote_nodemap = None
        
        # State tracking
        self.is_acquiring = False
        self.is_initialized = False
        
        # Hot pixel correction object (created when needed)
        self._hotpixel_correction_obj = None
        
        # Metadata state variables (for advanced mode)
        self.init_latitude = None
        self.init_longitude = None
        self.measured_alt = None
        self.measured_azi = None
        self.measured_lat = None
        self.measured_lon = None
        self.head_diff = None
        self.gps_heading_diff = None
        
        # Thread synchronization
        self.state_lock = threading.RLock()
        
        # Validate IDS Peak availability
        if not HAS_IDS_PEAK:
            logger.error("IDS Peak libraries not available. Camera operations will fail.")
    
    def get_api_version(self) -> str:
        """Get IDS Peak API version information."""
        try:
            version_obj = getattr(ids_peak, 'Version', None)
            if version_obj:
                return str(version_obj)
            return "Unknown"
        except:
            return "Unknown"
    
    def setup_device(self) -> bool:
        """Initialize the library and open the first available device."""
        if not HAS_IDS_PEAK:
            logger.error("IDS Peak libraries not available")
            return False
            
        try:
            # Initialize library (handle already initialized case)
            try:
                ids_peak.Library.Initialize()
                logger.info("IDS Peak library initialized")
            except ids_peak.Exception as e:
                if "Library already initialized" not in str(e):
                    raise
                else:
                    logger.warning("IDS Peak library was already initialized")
            
            # Setup device manager
            device_manager = ids_peak.DeviceManager.Instance()
            device_manager.Update()
            
            if device_manager.Devices().empty():
                logger.error("No camera device found")
                return False
            
            # Open first available device
            self.device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
            logger.info(f"Device opened: {self.device.ModelName()}")
            
            # Get nodemap
            self.remote_nodemap = self.device.RemoteDevice().NodeMaps()[0]
            
            return True
            
        except ids_peak.Exception as e:
            logger.error(f"IDS Peak exception during device setup: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during device setup: {e}")
            return False
    
    def configure_device(self) -> bool:
        """Configure device settings (exposure, pixel format, gain, etc.)."""
        if not self.device or not self.remote_nodemap:
            logger.error("Device not available for configuration")
            return False
            
        try:
            # Load default user set (from simpleExposure approach)
            try:
                self.remote_nodemap.FindNode("UserSetSelector").SetCurrentEntry("Default")
                self.remote_nodemap.FindNode("UserSetLoad").Execute()
                self.remote_nodemap.FindNode("UserSetLoad").WaitUntilDone()
                logger.info("Default user set loaded")
            except ids_peak.Exception as e:
                logger.warning(f"Could not load default user set: {e}")
            
            # Configure exposure time with validation
            exposure_time_node = self.remote_nodemap.FindNode("ExposureTime")
            min_exp, max_exp = exposure_time_node.Minimum(), exposure_time_node.Maximum()
            target_exp_us = self.exposure_time_ms * 1000  # Convert to microseconds
            
            if min_exp <= target_exp_us <= max_exp:
                exposure_time_node.SetValue(target_exp_us)
                logger.info(f"Exposure time set to {self.exposure_time_ms} ms ({target_exp_us} µs)")
            else:
                clamped_exp = max(min_exp, min(target_exp_us, max_exp))
                exposure_time_node.SetValue(clamped_exp)
                actual_ms = clamped_exp / 1000
                logger.warning(f"Exposure time clamped from {self.exposure_time_ms} ms to {actual_ms:.3f} ms")
                self.exposure_time_ms = actual_ms
            
            # Configure gain
            try:
                gain_auto_node = self.remote_nodemap.FindNode("GainAuto")
                if gain_auto_node:
                    if self.enable_auto_gain:
                        gain_auto_node.SetCurrentEntry("Continuous")
                        logger.info("Auto gain enabled")
                    else:
                        gain_auto_node.SetCurrentEntry("Off")
                        logger.info("Auto gain disabled")
            except ids_peak.Exception as e:
                logger.warning(f"Could not configure gain: {e}")
            
            # Set pixel format
            try:
                self.remote_nodemap.FindNode("PixelFormat").SetCurrentEntry('Mono12')
                logger.info("Pixel format set to Mono12")
            except ids_peak.Exception as e:
                logger.error(f"Failed to set pixel format: {e}")
                return False
            
            # Lock transport layer parameters
            try:
                self.remote_nodemap.FindNode("TLParamsLocked").SetValue(1)
                logger.info("Transport layer parameters locked")
            except ids_peak.Exception as e:
                logger.error(f"Failed to lock TL parameters: {e}")
                return False
            
            return True
            
        except ids_peak.Exception as e:
            logger.error(f"IDS Peak exception during configuration: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during configuration: {e}")
            return False
    
    def allocate_buffers(self) -> bool:
        """Allocate and queue buffers for image acquisition."""
        if not self.device or not self.remote_nodemap:
            logger.error("Device not available for buffer allocation")
            return False
            
        try:
            # Check if data stream already exists
            if self.data_stream:
                logger.warning("Data stream already exists, skipping allocation")
                return True
            
            # Open data stream
            stream_infos = self.device.DataStreams()
            if not stream_infos:
                logger.error("No data streams found on device")
                return False
                
            self.data_stream = stream_infos[0].OpenDataStream()
            logger.info("Data stream opened")
            
            # Calculate buffer requirements
            payload_size = self.remote_nodemap.FindNode("PayloadSize").Value()
            buffer_count = self.num_buffers or self.data_stream.NumBuffersAnnouncedMinRequired()
            
            # Clean up any existing buffers
            announced_buffers = self.data_stream.AnnouncedBuffers()
            if announced_buffers:
                logger.info(f"Cleaning up {len(announced_buffers)} existing buffers")
                self.data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
                for buffer in announced_buffers:
                    try:
                        self.data_stream.RevokeBuffer(buffer)
                    except ids_peak.Exception as e:
                        logger.warning(f"Could not revoke existing buffer: {e}")
            
            # Allocate new buffers
            for i in range(buffer_count):
                buffer = self.data_stream.AllocAndAnnounceBuffer(payload_size)
                self.data_stream.QueueBuffer(buffer)
                logger.debug(f"Allocated buffer {i+1}/{buffer_count}")
            
            logger.info(f"Allocated and queued {buffer_count} buffers")
            return True
            
        except ids_peak.Exception as e:
            logger.error(f"IDS Peak exception during buffer allocation: {e}")
            # Clean up on failure
            if self.data_stream:
                try:
                    self.data_stream = None
                except:
                    pass
            return False
        except Exception as e:
            logger.error(f"Unexpected error during buffer allocation: {e}")
            if self.data_stream:
                self.data_stream = None
            return False
    
    def initialize(self) -> bool:
        """
        Complete camera initialization sequence.
        Call this before starting any acquisition.
        """
        if self.is_initialized:
            logger.info("Camera already initialized")
            return True
            
        logger.info("Initializing camera system...")
        
        if not self.setup_device():
            logger.error("Device setup failed")
            return False
            
        if not self.configure_device():
            logger.error("Device configuration failed")
            return False
            
        if not self.allocate_buffers():
            logger.error("Buffer allocation failed")
            return False
        
        # Initialize hot pixel correction if enabled
        if self.hotpixel_correction:
            try:
                self._hotpixel_correction_obj = ids_peak_ipl.HotpixelCorrection()
                logger.info("Hot pixel correction initialized")
            except Exception as e:
                logger.warning(f"Could not initialize hot pixel correction: {e}")
                self.hotpixel_correction = False
        
        self.is_initialized = True
        logger.info("Camera initialization complete")
        return True
    
    def start_acquisition(self) -> bool:
        """Start camera acquisition."""
        if not self.is_initialized:
            logger.error("Camera not initialized. Call initialize() first.")
            return False
            
        if self.is_acquiring:
            logger.warning("Acquisition already started")
            return True
            
        try:
            # Start data stream acquisition
            self.data_stream.StartAcquisition()
            
            # Start camera acquisition
            self.remote_nodemap.FindNode("AcquisitionStart").Execute()
            self.remote_nodemap.FindNode("AcquisitionStart").WaitUntilDone(1000)
            
            self.is_acquiring = True
            logger.info("Camera acquisition started")
            return True
            
        except ids_peak.Exception as e:
            logger.error(f"IDS Peak exception starting acquisition: {e}")
            self.is_acquiring = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error starting acquisition: {e}")
            self.is_acquiring = False
            return False
    
    def stop_acquisition(self) -> bool:
        """Stop camera acquisition."""
        if not self.is_acquiring:
            logger.info("Acquisition not running")
            return True
            
        try:
            # Stop camera acquisition
            acq_stop_node = self.remote_nodemap.FindNode("AcquisitionStop")
            if acq_stop_node and acq_stop_node.IsAvailable():
                acq_stop_node.Execute()
                acq_stop_node.WaitUntilDone(1000)
            
            # Stop data stream
            if self.data_stream:
                try:
                    self.data_stream.StopAcquisition()
                except ids_peak.Exception as e:
                    if "Invalid parameter" in str(e):
                        self.data_stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
                    else:
                        raise
            
            self.is_acquiring = False
            logger.info("Camera acquisition stopped")
            return True
            
        except ids_peak.Exception as e:
            logger.error(f"IDS Peak exception stopping acquisition: {e}")
            self.is_acquiring = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error stopping acquisition: {e}")
            self.is_acquiring = False
            return False
    
    def acquire_single_image(self, timeout_ms: int = 5000) -> Tuple[Optional[np.ndarray], Optional[Any]]:
        """
        Acquire a single image (advanced mode for continuous operation).
        
        Returns:
            Tuple of (image_data, buffer) or (None, None) on failure
        """
        if not self.is_acquiring:
            logger.error("Acquisition not started")
            return None, None
            
        try:
            # Wait for buffer
            buffer = self.data_stream.WaitForFinishedBuffer(timeout_ms)
            if buffer is None:
                logger.warning("No buffer received within timeout")
                return None, None
            
            # Check buffer status
            if hasattr(buffer, "IsIncomplete") and buffer.IsIncomplete():
                logger.warning("Received incomplete buffer, re-queuing")
                self.queue_buffer(buffer)
                return None, None
            
            if hasattr(buffer, "HasNewData") and not buffer.HasNewData():
                logger.warning("Buffer has no new data, re-queuing")
                self.queue_buffer(buffer)
                return None, None
            
            # Convert buffer to image
            img = ids_peak_ipl_extension.BufferToImage(buffer)
            
            # Apply hot pixel correction if enabled
            if self.hotpixel_correction and self._hotpixel_correction_obj:
                img = self._hotpixel_correction_obj.CorrectAdaptive(img)
            
            # Get numpy array (make a copy for safety)
            image_data = img.get_numpy_2D_16().copy()
            
            logger.debug(f"Acquired image with shape {image_data.shape}")
            return image_data, buffer
            
        except ids_peak.Exception as e:
            if "Timeout" in str(e):
                logger.warning(f"Timeout waiting for image ({timeout_ms}ms)")
            else:
                logger.error(f"IDS Peak exception acquiring image: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error acquiring image: {e}")
            return None, None
    
    def queue_buffer(self, buffer: Any) -> bool:
        """Queue a buffer back to the data stream."""
        if not self.data_stream or not buffer:
            return False
            
        try:
            # Check if buffer is already queued
            if hasattr(buffer, "IsQueued") and buffer.IsQueued():
                logger.debug("Buffer already queued")
                return True
                
            self.data_stream.QueueBuffer(buffer)
            logger.debug("Buffer queued successfully")
            return True
            
        except ids_peak.Exception as e:
            logger.error(f"IDS Peak exception queueing buffer: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error queueing buffer: {e}")
            return False
    
    def process_batch_images(self, num_images: int, timeout_ms: Optional[int] = None) -> bool:
        """
        Process a batch of images (simple mode like simpleExposure).
        
        Args:
            num_images: Number of images to capture
            timeout_ms: Timeout for each image (calculated from exposure if None)
        """
        if not self.is_initialized:
            logger.error("Camera not initialized")
            return False
        
        if not self.start_acquisition():
            logger.error("Could not start acquisition")
            return False
        
        # Calculate timeout
        if timeout_ms is None:
            timeout_ms = max(int(self.exposure_time_ms + 2000), 5000)
        
        logger.info(f"Starting batch acquisition of {num_images} images")
        logger.info(f"Using timeout of {timeout_ms} ms per image")
        
        success_count = 0
        
        try:
            for i in range(num_images):
                try:
                    # Get image
                    buffer = self.data_stream.WaitForFinishedBuffer(timeout_ms)
                    img = ids_peak_ipl_extension.BufferToImage(buffer)
                    
                    # Apply hot pixel correction if enabled
                    if self.hotpixel_correction and self._hotpixel_correction_obj:
                        img = self._hotpixel_correction_obj.CorrectAdaptive(img)
                    
                    # Get image data
                    image_data = img.get_numpy_2D_16()
                    
                    # Save image
                    if self.save_fits(image_data, i + 1):
                        success_count += 1
                        
                    if self.enable_jpeg_output:
                        self.save_jpeg(image_data, i + 1)
                    
                    # Re-queue buffer
                    self.data_stream.QueueBuffer(buffer)
                    
                    logger.info(f"Processed image {i + 1}/{num_images}")
                    
                except Exception as e:
                    logger.error(f"Error processing image {i + 1}: {e}")
                    
        finally:
            self.stop_acquisition()
        
        logger.info(f"Batch acquisition complete: {success_count}/{num_images} images saved")
        return success_count == num_images
    
    def save_fits(self, image_data: np.ndarray, exposure_num: int, 
                  gps_lat: Optional[float] = None, gps_lon: Optional[float] = None) -> bool:
        """Save image data to FITS file with metadata."""
        if image_data is None:
            logger.error("Cannot save FITS: image data is None")
            return False
        
        try:
            # Create output directory
            fits_dir = os.path.join(self.output_dir, "fits")
            os.makedirs(fits_dir, exist_ok=True)
            
            # Generate filename
            now = datetime.now()
            filename = now.strftime(f"image_%Y%m%d_%H_%M_%S_{exposure_num:03d}.fits")
            filepath = os.path.join(fits_dir, filename)
            
            # Create FITS HDU
            hdu = fits.PrimaryHDU(image_data.astype(np.uint16))
            hdr = hdu.header
            
            # Basic metadata
            hdr['DATE-OBS'] = (now.isoformat(), 'Timestamp of observation start')
            hdr['EXPTIME'] = (self.exposure_time_ms, 'Exposure time in milliseconds')
            hdr['IMG_NUM'] = (exposure_num, 'Image sequence number')
            
            # GPS metadata (thread-safe access)
            with self.state_lock:
                if gps_lat is not None:
                    hdr['GPS_LAT'] = (gps_lat, 'GPS latitude at capture time')
                elif self.init_latitude is not None:
                    hdr['GPS_LAT'] = (self.init_latitude, 'GPS latitude (last known)')
                    
                if gps_lon is not None:
                    hdr['GPS_LON'] = (gps_lon, 'GPS longitude at capture time')
                elif self.init_longitude is not None:
                    hdr['GPS_LON'] = (self.init_longitude, 'GPS longitude (last known)')
                
                # Analysis metadata (if available)
                if self.measured_lat is not None:
                    hdr['MEAS_LAT'] = (self.measured_lat, 'Calculated latitude from sun analysis')
                if self.measured_lon is not None:
                    hdr['MEAS_LON'] = (self.measured_lon, 'Calculated longitude from sun analysis')
                if self.measured_alt is not None:
                    hdr['MEAS_ALT'] = (self.measured_alt, 'Measured sun altitude (degrees)')
                if self.measured_azi is not None:
                    hdr['MEAS_AZI'] = (self.measured_azi, 'Measured sun azimuth (degrees)')
                if self.head_diff is not None:
                    hdr['HEAD_DIF'] = (self.head_diff, 'Heading difference (calculated - measured)')
            
            # Write file
            hdu.writeto(filepath, overwrite=True)
            logger.info(f"Saved FITS file: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save FITS file: {e}", exc_info=True)
            return False
    
    def save_jpeg(self, image_data: np.ndarray, exposure_num: int,
                  edges: Optional[np.ndarray] = None, 
                  sun_location: Optional[Tuple[float, float]] = None) -> bool:
        """Save image data to JPEG file with optional overlays."""
        if image_data is None:
            logger.error("Cannot save JPEG: image data is None")
            return False
        
        try:
            import cv2
            
            # Create output directory
            jpeg_dir = os.path.join(self.output_dir, "jpeg")
            os.makedirs(jpeg_dir, exist_ok=True)
            
            # Generate filename
            now = datetime.now()
            filename = now.strftime(f"image_%Y%m%d_%H_%M_%S_{exposure_num:03d}.jpg")
            filepath = os.path.join(jpeg_dir, filename)
            
            # Normalize image to 8-bit
            img_min, img_max = np.min(image_data), np.max(image_data)
            img_normalized = np.clip((image_data - img_min) / (img_max - img_min) * 255, 0, 255).astype(np.uint8)
            
            # Apply CLAHE enhancement
            clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
            img_enhanced = clahe.apply(img_normalized)
            
            # Convert to BGR for overlay drawing
            img_bgr = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2BGR)
            
            # Flip image if needed (for allsky camera orientation)
            img_bgr = cv2.flip(img_bgr, 1)
            
            # Draw circle overlay if edges provided
            if edges is not None and len(edges) > 0:
                try:
                    x, y, radius = int(edges[0, 0]), int(edges[0, 1]), int(edges[0, 2])
                    height, width = img_bgr.shape[:2]
                    
                    # Transform coordinates for flipped image
                    transformed_x = x
                    transformed_y = height - 1 - y
                    
                    # Draw circle and center point
                    cv2.circle(img_bgr, (transformed_x, transformed_y), radius, (0, 255, 255), 2)  # Yellow circle
                    cv2.circle(img_bgr, (transformed_x, transformed_y), 5, (255, 0, 0), -1)  # Blue center
                    
                    logger.debug(f"Drew circle overlay at ({transformed_x}, {transformed_y}), radius {radius}")
                except Exception as e:
                    logger.warning(f"Could not draw circle overlay: {e}")
            
            # Draw sun location if provided
            if sun_location is not None and len(sun_location) >= 2:
                try:
                    sun_x, sun_y = int(sun_location[0]), int(sun_location[1])
                    height, width = img_bgr.shape[:2]
                    
                    # Transform coordinates
                    transformed_sun_x = sun_x
                    transformed_sun_y = height - 1 - sun_y
                    
                    # Draw X mark
                    x_size, thickness = 15, 3
                    color = (0, 0, 255)  # Red
                    
                    cv2.line(img_bgr,
                            (transformed_sun_x - x_size, transformed_sun_y - x_size),
                            (transformed_sun_x + x_size, transformed_sun_y + x_size),
                            color, thickness)
                    cv2.line(img_bgr,
                            (transformed_sun_x - x_size, transformed_sun_y + x_size),
                            (transformed_sun_x + x_size, transformed_sun_y - x_size),
                            color, thickness)
                    
                    logger.debug(f"Drew sun location X at ({transformed_sun_x}, {transformed_sun_y})")
                except Exception as e:
                    logger.warning(f"Could not draw sun location: {e}")
            
            # Save image
            cv2.imwrite(filepath, img_bgr)
            logger.info(f"Saved JPEG file: {filepath}")
            return True
            
        except ImportError:
            logger.warning("OpenCV not available, skipping JPEG output")
            return False
        except Exception as e:
            logger.error(f"Failed to save JPEG file: {e}", exc_info=True)
            return False
    
    def update_gps_coordinates(self, latitude: float, longitude: float):
        """Update GPS coordinates (thread-safe)."""
        with self.state_lock:
            self.init_latitude = latitude
            self.init_longitude = longitude
    
    def update_analysis_results(self, measured_alt: Optional[float] = None,
                               measured_azi: Optional[float] = None,
                               measured_lat: Optional[float] = None,
                               measured_lon: Optional[float] = None,
                               head_diff: Optional[float] = None):
        """Update analysis results (thread-safe)."""
        with self.state_lock:
            if measured_alt is not None:
                self.measured_alt = measured_alt
            if measured_azi is not None:
                self.measured_azi = measured_azi
            if measured_lat is not None:
                self.measured_lat = measured_lat
            if measured_lon is not None:
                self.measured_lon = measured_lon
            if head_diff is not None:
                self.head_diff = head_diff
    
    def cleanup(self):
        """Clean up all camera resources."""
        logger.info("Starting camera cleanup...")
        
        # Stop acquisition
        if self.is_acquiring:
            logger.info("Stopping acquisition...")
            self.stop_acquisition()
        
        # Clean up data stream
        if self.data_stream:
            try:
                logger.info("Cleaning up data stream...")
                
                # Flush buffers
                try:
                    self.data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
                except ids_peak.Exception as e:
                    if "Acquisition running" not in str(e):
                        logger.warning(f"Error flushing data stream: {e}")
                
                # Revoke buffers
                announced_buffers = self.data_stream.AnnouncedBuffers()
                for buffer in announced_buffers:
                    try:
                        self.data_stream.RevokeBuffer(buffer)
                    except ids_peak.Exception as e:
                        logger.warning(f"Could not revoke buffer: {e}")
                
                logger.info(f"Attempted to revoke {len(announced_buffers)} buffers")
                
            except Exception as e:
                logger.error(f"Error during data stream cleanup: {e}")
            finally:
                self.data_stream = None
        
        # Unlock transport layer parameters
        if self.remote_nodemap:
            try:
                tl_locked_node = self.remote_nodemap.FindNode("TLParamsLocked")
                if tl_locked_node and tl_locked_node.IsAvailable() and tl_locked_node.Value() == 1:
                    logger.info("Unlocking transport layer parameters...")
                    tl_locked_node.SetValue(0)
            except Exception as e:
                logger.warning(f"Could not unlock TL parameters: {e}")
        
        # Clear device reference
        if self.device:
            logger.info("Clearing device reference...")
            self.device = None
        
        # Close library
        try:
            logger.info("Closing IDS Peak library...")
            ids_peak.Library.Close()
            logger.info("IDS Peak library closed")
        except ids_peak.Exception as e:
            if "Library not initialized" not in str(e):
                logger.error(f"Error closing IDS Peak library: {e}")
        except Exception as e:
            logger.error(f"Unexpected error closing library: {e}")
        
        # Reset state
        self.is_initialized = False
        self.is_acquiring = False
        self.remote_nodemap = None
        self._hotpixel_correction_obj = None
        
        logger.info("Camera cleanup complete")
    
    def __enter__(self):
        """Context manager entry."""
        if not self.initialize():
            raise RuntimeError("Failed to initialize camera")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Convenience functions for different use cases

def create_simple_camera(exposure_time_ms: float = 10.0, 
                        output_dir: str = "output",
                        hotpixel_correction: bool = False) -> CameraAcquisition:
    """Create a camera instance configured for simple batch processing."""
    return CameraAcquisition(
        exposure_time_ms=exposure_time_ms,
        output_dir=output_dir,
        hotpixel_correction=hotpixel_correction,
        enable_jpeg_output=False,
        enable_auto_gain=False
    )

def create_advanced_camera(exposure_time_ms: float = 0.02,
                          output_dir: str = "output",
                          enable_jpeg_output: bool = True) -> CameraAcquisition:
    """Create a camera instance configured for advanced continuous operation."""
    return CameraAcquisition(
        exposure_time_ms=exposure_time_ms,
        output_dir=output_dir,
        hotpixel_correction=False,
        enable_jpeg_output=enable_jpeg_output,
        enable_auto_gain=False
    )
