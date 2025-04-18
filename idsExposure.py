# /home/myuser/AllSkyCam/idsExposure.py
import logging
import os
from logger_config import setup_logging
setup_logging()

program_name = os.path.basename(__file__)
logger = logging.getLogger(program_name)

try:
    from ids_peak import ids_peak
    from ids_peak import ids_peak_ipl_extension
    HAS_IDS_PEAK = True
except ImportError:
    logger.error("Failed to import IDS Peak libraries. Please ensure they are installed correctly.")
    HAS_IDS_PEAK = False
    # Define placeholders so the rest of the code doesn't error on import
    class ids_peak:
        class DataStreamFlushMode_DiscardAll: pass
        class DeviceAccessType_Control: pass
        class Library:
            @staticmethod
            def Initialize(): pass
            @staticmethod
            def Close(): pass

import argparse
import logging
from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension
import numpy as np
from astropy.io import fits
import os
from datetime import datetime
from detectSun import ImageProcessor # Assuming this class is correctly defined
from ubloxReader import GPSReader # Assuming this class is correctly defined
from gpsHandler import GPSHandler # Import our new GPS Handler class
import time
import threading
import math
import tkinter as tk
import gc # For garbage collection during shutdown

program_name = os.path.basename(__file__)
logger = logging.getLogger(program_name)

# --- HeadingVisualizer class remains the same ---
class HeadingVisualizer:
    """Class to visualize the heading direction in real-time using tkinter."""
    def __init__(self, root):
        self.heading = 0  # Initial heading value
        self.root = root
        self.root.title("Heading Visualizer")
        self.canvas = tk.Canvas(self.root, width=600, height=600, bg="white")
        self.canvas.pack()

        # Draw a circle to represent the compass
        self.center_x = 300
        self.center_y = 300
        self.radius = 250
        self.canvas.create_oval(
            self.center_x - self.radius, self.center_y - self.radius,
            self.center_x + self.radius, self.center_y + self.radius,
            outline="black"
        )

        # Draw the initial arrow
        self.arrow = self.canvas.create_line(
            self.center_x, self.center_y,
            self.center_x, self.center_y - self.radius,
            arrow=tk.LAST, fill="blue", width=10, arrowshape=(20, 25, 10)
        )

        # Add heading text
        self.heading_text = self.canvas.create_text(300, 500, text="Heading: 0°")

    def update_heading(self, heading):
        """Update the heading direction and text."""
        # Ensure GUI updates happen in the main thread
        if self.root and self.root.winfo_exists():
             self.root.after(0, self._update_heading_safe, heading)

    def _update_heading_safe(self, heading):
        """Safely update heading and text from the main thread."""
        if self.root and self.root.winfo_exists():
            self.heading = heading
            self._update_arrow()
            self._update_heading_text()

    def _update_arrow(self):
        """Update the arrow direction on the canvas."""
        if self.canvas and self.canvas.winfo_exists():
            angle_rad = math.radians(self.heading)
            end_x = self.center_x + self.radius * math.sin(angle_rad)
            end_y = self.center_y - self.radius * math.cos(angle_rad)
            self.canvas.coords(
                self.arrow,
                self.center_x, self.center_y,  # Start point
                end_x, end_y  # End point
            )

    def _update_heading_text(self):
        """Update the heading text display."""
        if self.canvas and self.canvas.winfo_exists():
            self.canvas.itemconfig(self.heading_text, text=f"Heading: {self.heading:.1f}°")

# --- CameraAcquisition class (camera hardware interaction only) ---
class CameraAcquisition:
    """
    Manages IDS Peak camera setup, control, single image acquisition,
    metadata storage, and resource cleanup. The main acquisition loop
    is handled externally.
    """
    def __init__(self, exposure_time_ms=0.02, num_buffers=None,
                 output_dir="output"):

        self.exposure_time_ms = exposure_time_ms
        self.num_buffers = num_buffers
        self.output_dir = output_dir

        self.device = None
        self.data_stream = None
        self.remote_nodemap = None
        self.is_acquiring = False # Flag to track acquisition state

        # --- State variables for metadata (updated externally) ---
        # Note: GPS coordinates are now handled by GPSHandler
        self.init_latitute = None  # Will be populated from GPSHandler
        self.init_longitude = None # Will be populated from GPSHandler
        self.measured_alt = None
        self.measured_azi = None
        self.measured_lat = None
        self.measured_lon = None
        self.head_diff = None
        self.gpsHeadingDiff = None
        # --- End State variables ---

        # Thread synchronization for state variables
        self.state_lock = threading.RLock()

    def _get_api_version(self):
        """Helper method to get API version details and adapt method calls."""
        try:
            version_obj = getattr(ids_peak, 'Version', None)
            if version_obj:
                return str(version_obj)
            return "Unknown"
        except:
            return "Unknown"

    # Remove GPS-related methods
    # start_gps_thread, stop_gps_thread, update_gps_location are removed

    def setup_device(self):
        """Initializes the library and opens the first available device."""
        try:
            ids_peak.Library.Initialize()
        except ids_peak.Exception as e:
            logger.error(f"Failed to initialize IDS Peak library: {e}")
            if "Library already initialized" not in str(e):
                 raise # Re-raise if it's a different error
            else:
                 logger.warning("IDS Peak library was already initialized.")

        device_manager = ids_peak.DeviceManager.Instance()
        device_manager.Update()

        if device_manager.Devices().empty():
            logger.error("No device found. Exiting program.")
            # No need to close library here, cleanup handles it
            return False

        try:
            self.device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
            logger.info(f"Device opened: {self.device.ModelName()}")
            self.remote_nodemap = self.device.RemoteDevice().NodeMaps()[0] # Get nodemap here
            return True
        except ids_peak.Exception as e:
            logger.error(f"Failed to open device: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error opening device: {e}")
            return False

    def configure_device(self):
        """Configures the device settings (exposure, pixel format, TLParams)."""
        if not self.device or not self.remote_nodemap:
            logger.error("Device not open or nodemap not available, cannot configure.")
            return False
        try:
            # --- Set Exposure Time ---
            exposure_time_node = self.remote_nodemap.FindNode("ExposureTime")
            min_exp, max_exp = exposure_time_node.Minimum(), exposure_time_node.Maximum()
            target_exp_us = self.exposure_time_ms * 1000
            if min_exp <= target_exp_us <= max_exp:
                exposure_time_node.SetValue(target_exp_us)
                logger.info(f"Exposure time set to {self.exposure_time_ms} ms ({target_exp_us} us)")
            else:
                clamped_exp = max(min_exp, min(target_exp_us, max_exp))
                exposure_time_node.SetValue(clamped_exp)
                logger.warning(f"Requested exposure {target_exp_us} is outside range [{min_exp}, {max_exp}]. Clamped to {clamped_exp} us.")

            # --- Set Pixel Format ---
            try:
                self.remote_nodemap.FindNode("PixelFormat").SetCurrentEntry('Mono12')
                logger.info("Pixel format set to Mono12")
            except ids_peak.Exception as e:
                 logger.error(f"Failed to set PixelFormat to Mono12: {e}. Check camera support.")
                 return False # Pixel format is critical

            # --- Lock Transport Layer Parameters (Important!) ---
            try:
                self.remote_nodemap.FindNode("TLParamsLocked").SetValue(1)
                logger.info("Transport layer parameters locked.")
            except ids_peak.Exception as e:
                logger.error(f"Failed to lock TLParams: {e}")
                return False # Locking is important for stability

            return True

        except ids_peak.Exception as e:
             logger.error(f"IDS Peak Exception during configuration: {e}")
             return False
        except Exception as e:
            logger.error(f"General error configuring device: {e}")
            return False

    def allocate_buffers(self):
        """Opens the data stream and allocates/queues buffers."""
        if not self.device or not self.remote_nodemap:
            logger.error("Device not open or nodemap not available, cannot allocate buffers.")
            return False
        try:
            # Check if datastream already open (e.g., during restart attempts)
            if self.data_stream:
                 logger.warning("Data stream already seems to be open. Skipping allocation.")
                 return True

            stream_infos = self.device.DataStreams()
            if not stream_infos:
                 logger.error("No data streams found on the device.")
                 return False

            self.data_stream = stream_infos[0].OpenDataStream()
            logger.info("Data stream opened.")

            payload_size = self.remote_nodemap.FindNode("PayloadSize").Value()
            buffer_count = self.num_buffers or self.data_stream.NumBuffersAnnouncedMinRequired()

            # Revoke any existing buffers first (important for clean state)
            announced_buffers = self.data_stream.AnnouncedBuffers()
            if announced_buffers:
                logger.warning(f"Revoking {len(announced_buffers)} existing buffers before allocation.")
                self.data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
                for buffer in announced_buffers:
                    try:
                        self.data_stream.RevokeBuffer(buffer)
                    except ids_peak.Exception as buf_err:
                        logger.warning(f"Could not revoke existing buffer: {buf_err}")

            # Allocate and queue new buffers
            for i in range(buffer_count):
                buffer = self.data_stream.AllocAndAnnounceBuffer(payload_size)
                self.data_stream.QueueBuffer(buffer)
                logger.debug(f"Allocated and queued buffer {i+1}/{buffer_count}")

            logger.info(f"Allocated and queued {buffer_count} buffers.")
            return True
        except ids_peak.Exception as e:
             logger.error(f"IDS Peak Exception allocating buffers: {e}")
             # Clean up partially opened stream if allocation fails
             if self.data_stream:
                 try: self.data_stream.Close()
                 except: pass
                 self.data_stream = None
             return False
        except Exception as e:
            logger.error(f"General error allocating buffers: {e}")
            if self.data_stream:
                 try: self.data_stream.Close()
                 except: pass
                 self.data_stream = None
            return False

    def start_acquisition(self):
        """Starts the camera acquisition process."""
        if not self.data_stream or not self.remote_nodemap:
            logger.error("Data stream or nodemap not available. Cannot start acquisition.")
            return False
        if self.is_acquiring:
            logger.warning("Acquisition already started.")
            return True
            
        # Track success stages for robust error handling
        acquisition_started = False
            
        try:
            # FIXED: Use StartAcquisition without parameters according to the API version
            self.data_stream.StartAcquisition()  # Remove the timeout parameter
            acquisition_started = True
            
            self.remote_nodemap.FindNode("AcquisitionStart").Execute()
            self.remote_nodemap.FindNode("AcquisitionStart").WaitUntilDone(1000)  # Wait briefly
            self.is_acquiring = True
            logger.info("Camera acquisition started.")
            return True
        except ids_peak.Exception as e:
             logger.error(f"IDS Peak Exception starting acquisition: {e}")
             # If we started acquisition but couldn't complete fully, try to stop it
             if acquisition_started:
                 try:
                     self.data_stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
                 except Exception as stop_err:
                     logger.warning(f"Error stopping acquisition after failed start: {stop_err}")
             self.is_acquiring = False  # Ensure flag is false on failure
             return False
        except Exception as e:
            logger.error(f"General error starting acquisition: {e}")
            if acquisition_started:
                try:
                    self.data_stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
                except:
                    pass
            self.is_acquiring = False
            return False

    def stop_acquisition(self):
        """Stops the camera acquisition process."""
        if not self.is_acquiring:
            logger.info("Acquisition not running or already stopped.")
            return True # Considered success if not running

        stopped_cleanly = False
        try:
            # Check if nodes exist before trying to execute
            acq_stop_node = self.remote_nodemap.FindNode("AcquisitionStop")
            if acq_stop_node and acq_stop_node.IsAvailable():
                 acq_stop_node.Execute()
                 acq_stop_node.WaitUntilDone(1000) # Wait briefly
            else:
                 logger.warning("AcquisitionStop node not available or executable.")

            if self.data_stream:
                try:
                    self.data_stream.StopAcquisition()  # Try without parameters first
                except ids_peak.Exception as e:
                    if "Invalid parameter" in str(e):
                        # If we get parameter error, try with the mode parameter
                        logger.info("Trying stop acquisition with AcquisitionStopMode_Default...")
                        self.data_stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
                    else:
                        raise
            else:
                 logger.warning("Data stream not available to stop acquisition.")

            self.is_acquiring = False
            stopped_cleanly = True
            logger.info("Camera acquisition stopped.")
        except ids_peak.Exception as e:
             logger.error(f"IDS Peak Exception stopping acquisition: {e}")
             # Attempt to force flag to false even on error
             self.is_acquiring = False
        except Exception as e:
            logger.error(f"General error stopping acquisition: {e}")
            self.is_acquiring = False
        finally:
            # Return status based on whether the flag was successfully set to False
            return stopped_cleanly

    def acquire_image(self, timeout_ms=2000):
        """
        Acquires a single image buffer.
        """
        if not self.is_acquiring or not self.data_stream:
            logger.error("Acquisition not started or stream unavailable. Cannot acquire image.")
            return None, None
        try:
            # Use explicit timeout value
            logger.debug(f"Waiting for buffer with {timeout_ms}ms timeout")
            buffer = self.data_stream.WaitForFinishedBuffer(timeout_ms)
            if buffer is None:
                logger.warning("WaitForFinishedBuffer returned None")
                return None, None
                
            # Check buffer status
            has_new_data = hasattr(buffer, "HasNewData") and buffer.HasNewData()
            is_incomplete = hasattr(buffer, "IsIncomplete") and buffer.IsIncomplete()
            
            if is_incomplete:
                 logger.warning(f"Acquired buffer is incomplete. Re-queuing.")
                 # Requeue incomplete buffer immediately
                 self.queue_buffer(buffer)
                 return None, None # Indicate failure

            if has_new_data:
                 img = ids_peak_ipl_extension.BufferToImage(buffer)
                 # Create a copy to avoid issues when buffer is requeued later
                 image_data = img.get_numpy_2D_16().copy()
                 logger.debug(f"Acquired image with shape {image_data.shape}")
                 # DO NOT QUEUE BUFFER HERE - return it to the caller
                 return image_data, buffer
            else:
                 # This case should be rare if IsIncomplete is checked first
                 logger.warning(f"Buffer received but contained no new data (timeout: {timeout_ms}ms). Re-queuing.")
                 self.queue_buffer(buffer) # Requeue empty/old buffer immediately
                 return None, None

        except ids_peak.Exception as e:
            if "Timeout" in str(e):
                logger.warning(f"Timeout waiting for finished buffer ({timeout_ms}ms).")
            else:
                logger.error(f"IDS Peak Exception acquiring image: {e}")
            return None, None # Indicate failure
        except Exception as e:
            logger.error(f"General error acquiring image: {e}")
            return None, None

    def queue_buffer(self, buffer):
        """Queues a buffer back to the data stream."""
        if not self.data_stream:
            logger.error("Data stream not available for queueing.")
            return False
        if not buffer:
            logger.error("Invalid buffer provided for queueing.")
            return False
            
        # Add checking if the buffer is valid for queueing
        try:
            # Check if buffer is already queued (if the API provides such a method)
            if hasattr(buffer, "IsQueued") and buffer.IsQueued():
                logger.warning("Buffer is already queued. Skipping.")
                return True
                
            self.data_stream.QueueBuffer(buffer)
            logger.debug("Buffer queued back.")
            return True
        except ids_peak.Exception as e:
            logger.error(f"IDS Peak Exception queueing buffer: {e}")
            return False
        except Exception as e:
            logger.error(f"General error queueing buffer: {e}")
            return False

    def save_fits(self, image_data, exposure_num, gps_lat=None, gps_lon=None):
        """
        Save the image data to a FITS file with metadata.
        """
        if image_data is None:
            logger.error("Cannot save FITS, image data is None.")
            return False # Indicate failure

        filepath = None # Initialize for error message
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            now = datetime.now()
            # Use microseconds for potentially higher frame rates
            filename = now.strftime(f"image_%Y%m%d_%H%M%S_{exposure_num:03d}.fits")
            filepath = os.path.join(self.output_dir, filename)

            hdu = fits.PrimaryHDU(image_data.astype(np.uint16)) # Ensure correct dtype for FITS
            hdr = hdu.header
            hdr['DATE-OBS'] = (now.isoformat(), 'Timestamp of observation start')
            hdr['EXPTIME'] = (self.exposure_time_ms, 'Exposure time in milliseconds')
            hdr['IMG_NUM'] = (exposure_num, 'Image sequence number in this run')

            # Thread-safe access to state variables
            with self.state_lock:
                # Prefer externally provided GPS data
                if gps_lat is not None:
                    hdr['GPS_LAT'] = (gps_lat, 'GPS latitude at capture time')
                elif self.init_latitute is not None:
                    hdr['GPS_LAT'] = (self.init_latitute, 'Last known GPS latitude')
                    
                if gps_lon is not None:
                    hdr['GPS_LON'] = (gps_lon, 'GPS longitude at capture time')
                elif self.init_longitude is not None:
                    hdr['GPS_LON'] = (self.init_longitude, 'Last known GPS longitude')
                
                # Add other metadata stored in the class attributes
                if self.measured_lat is not None:
                    hdr['MEAS_LAT'] = (self.measured_lat, 'Calculated Latitude (from Sun)')
                if self.measured_lon is not None:
                    hdr['MEAS_LON'] = (self.measured_lon, 'Calculated Longitude (from Sun)')
                if self.measured_alt is not None:
                    hdr['MEAS_ALT'] = (self.measured_alt, 'Calculated Sun Altitude (degrees)')
                if self.measured_azi is not None:
                    hdr['MEAS_AZI'] = (self.measured_azi, 'Calculated Sun Azimuth (degrees)')
                if self.head_diff is not None:
                    hdr['HEAD_DIF'] = (self.head_diff, 'Heading Diff (Sun Azi - Meas Azi)')

            hdu.writeto(filepath, overwrite=True)
            logger.info(f"Saved FITS file: {filepath}")
            return True # Indicate success

        except Exception as e:
            log_filepath = filepath if filepath else "unknown path"
            logger.error(f"Failed to save FITS file: {log_filepath}, error: {e}", exc_info=True)
            return False # Indicate failure

    def save_jpeg(self, image_data, exposure_num, edges=None, sun_location=None):
        """
        Save the image data to a JPEG file with optional circle overlay and sun location.
        
        Args:
            image_data: NumPy array containing the image data
            exposure_num: Exposure sequence number
            edges: Optional edge data containing circle parameters [x, y, radius]
            sun_location: Optional sun location (x, y, r) to mark with an X
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        if image_data is None:
            logger.error("Cannot save JPEG, image data is None.")
            return False

        filepath = None
        try:
            import cv2  # Import here to avoid dependency if not used
            
            # Create output directory if needed
            jpeg_dir = os.path.join(self.output_dir, "jpeg")
            os.makedirs(jpeg_dir, exist_ok=True)
            
            # Create filename
            now = datetime.now()
            filename = now.strftime(f"image_%Y%m%d_%H%M%S_{exposure_num:03d}.jpg")
            filepath = os.path.join(jpeg_dir, filename)
            
            # Normalize and convert to 8-bit for display
            img_min = np.min(image_data)
            img_max = np.max(image_data)
            img_normalized = np.clip((image_data - img_min) / (img_max - img_min) * 255, 0, 255).astype(np.uint8)
            
            # Create a black and white image (no colormap)
            # Convert to 3-channel so we can draw colored circles
            img_bw = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2BGR)
            
            # Draw circle if edges are provided
            if edges is not None and len(edges) > 0:
                # Extract circle parameters from edges
                try:
                    x, y, radius = int(edges[0, 0]), int(edges[0, 1]), int(edges[0, 2])
                    
                    # Transform coordinates to account for flipping done in analysis
                    # The edges were detected on flipped image (flip up-down and left-right)
                    height, width = img_bw.shape[:2]
                    transformed_x = width - 1 - x
                    transformed_y = height - 1 - y
                    
                    # Draw circle on the image
                    cv2.circle(img_bw, (transformed_x, transformed_y), radius, (0, 255, 255), 2)  # Yellow circle
                    # Draw center point
                    cv2.circle(img_bw, (transformed_x, transformed_y), 5, (255, 0, 0), -1)  # Blue center point
                    
                    logger.debug(f"Drew circle at ({transformed_x}, {transformed_y}) with radius {radius} on JPEG")
                    logger.debug(f"Original detected circle was at ({x}, {y})")
                except Exception as e:
                    logger.warning(f"Could not draw circle on image: {e}")
            
            # Draw sun location if provided
            if sun_location is not None and len(sun_location) >= 2:
                try:
                    sun_x, sun_y = int(sun_location[0]), int(sun_location[1])
                    
                    # Transform coordinates for sun position as well
                    height, width = img_bw.shape[:2]
                    transformed_sun_x = width - 1 - sun_x
                    transformed_sun_y = height - 1 - sun_y
                    
                    # Draw an X mark at the sun position
                    x_size = 15  # Size of the X mark
                    thickness = 3  # Line thickness
                    color = (0, 0, 255)  # Red color in BGR
                    
                    # Draw the X
                    cv2.line(img_bw, 
                              (transformed_sun_x - x_size, transformed_sun_y - x_size),
                              (transformed_sun_x + x_size, transformed_sun_y + x_size),
                              color, thickness)
                    cv2.line(img_bw, 
                              (transformed_sun_x - x_size, transformed_sun_y + x_size),
                              (transformed_sun_x + x_size, transformed_sun_y - x_size),
                              color, thickness)
                    
                    logger.debug(f"Drew X mark at sun location ({transformed_sun_x}, {transformed_sun_y}) on JPEG")
                    logger.debug(f"Original sun location was at ({sun_x}, {sun_y})")
                except Exception as e:
                    logger.warning(f"Could not draw sun location on image: {e}")
            
            # Save the image
            cv2.imwrite(filepath, img_bw)
            logger.info(f"Saved JPEG file: {filepath}")
            return True
            
        except Exception as e:
            log_filepath = filepath if filepath else "unknown path"
            logger.error(f"Failed to save JPEG file: {log_filepath}, error: {e}", exc_info=True)
            return False
    

    def cleanup(self):
        """Cleans up resources: stops acquisition, revokes buffers, closes stream/device/library."""
        logger.info("Starting camera cleanup...")

        # 1. Stop acquisition if running
        if self.is_acquiring:
            logger.info("Stopping acquisition as part of cleanup...")
            self.stop_acquisition()

        # Ensure acquisition is actually stopped before flushing
        if self.is_acquiring:
            logger.warning("Acquisition still appears to be running. Forcing flag to false.")
            self.is_acquiring = False
            time.sleep(0.5)  # Brief delay to ensure camera processes stop command

        # 2. Clean up DataStream (Flush, Revoke Buffers)
        if self.data_stream:
            try:
                logger.info("Flushing data stream...")
                try:
                    self.data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
                except ids_peak.Exception as flush_err:
                    if "Acquisition running" in str(flush_err):
                        logger.warning("Cannot flush - acquisition appears to still be running. Continuing cleanup.")
                    else:
                        raise

                logger.info("Revoking buffers...")
                announced_buffers = self.data_stream.AnnouncedBuffers()
                for buffer in announced_buffers:
                    try:
                        self.data_stream.RevokeBuffer(buffer)
                    except ids_peak.Exception as buf_err:
                        logger.warning(f"Could not revoke buffer during cleanup: {buf_err}")
                logger.info(f"Attempted to revoke {len(announced_buffers)} buffers.")

                # FIXED: The DataStream doesn't have a Close method in this IDS Peak version
                logger.info("Data stream will be cleared.")
                # self.data_stream.Close()  # Remove this line - no Close method

            except ids_peak.Exception as ds_err:
                logger.error(f"Error during data stream cleanup: {ds_err}")
            except Exception as e:
                logger.error(f"Unexpected error during data stream cleanup: {e}")
            finally:
                self.data_stream = None # Clear reference

        # 3. Unlock TLParams (Best effort)
        if self.remote_nodemap:
            try:
                tl_locked_node = self.remote_nodemap.FindNode("TLParamsLocked")
                if tl_locked_node and tl_locked_node.IsAvailable() and tl_locked_node.Value() == 1:
                    logger.info("Unlocking TLParams...")
                    tl_locked_node.SetValue(0)
            except ids_peak.Exception as tl_err:
                logger.error(f"Error unlocking TLParams during cleanup: {tl_err}")
            except Exception as e:
                logger.error(f"Unexpected error unlocking TLParams: {e}")
            # Don't clear nodemap reference here, device closing might need it implicitly

        # 4. Close Device (IDS Peak handles closing streams implicitly if not already done)
        # There's no explicit device.Close() in the Python API.
        # It's managed by the library closing or object destruction.
        if self.device:
             logger.info("Device object reference will be cleared.")
             self.device = None # Clear reference to allow garbage collection

        # 5. Close Library (This is the main cleanup step for the SDK)
        try:
            logger.info("Closing IDS Peak library...")
            ids_peak.Library.Close()
            logger.info("IDS Peak library closed.")
        except ids_peak.Exception as lib_err:
            # Avoid error if already closed
            if "Library not initialized" not in str(lib_err):
                 logger.error(f"Error closing IDS Peak library: {lib_err}")
        except Exception as e:
            logger.error(f"Unexpected error closing library: {e}")

        logger.info("Camera cleanup finished.")

    def initialize(self):
        """Initialize the camera system: setup, configure and allocate buffers."""
        if not HAS_IDS_PEAK:
            logger.error("IDS Peak libraries not available. Cannot initialize camera.")
            return False
            
        # Check IDS Peak version (optional)
        try:
            # This might need adjustment based on how IDS Peak reports its version
            version = getattr(ids_peak, 'Version', None)
            if version:
                logger.info(f"Using IDS Peak version: {version}")
        except:
            pass
            
        if not self.setup_device():
            logger.error("Failed to setup camera device")
            return False
            
        if not self.configure_device():
            logger.error("Failed to configure camera device")
            return False
            
        if not self.allocate_buffers():
            logger.error("Failed to allocate camera buffers")
            return False
            
        logger.info("Camera initialization complete")
        return True

    def log_api_capabilities(self):
        """Log available methods and their parameters for debugging."""
        try:
            # Log DataStream methods
            if self.data_stream:
                methods = dir(self.data_stream)
                logger.debug(f"Available DataStream methods: {', '.join(methods)}")
                
                # Check specific method signatures
                if hasattr(self.data_stream, "StartAcquisition"):
                    import inspect
                    sig = inspect.signature(self.data_stream.StartAcquisition)
                    logger.debug(f"StartAcquisition signature: {sig}")
        except:
            logger.warning("Failed to log API capabilities")


# NEW CLASS: Exposure sequence controller (separated from camera hardware)
class ExposureSequence:
    """
    Controls the exposure sequence, camera operation, and image processing.
    Keeps the acquisition loop separate from the camera hardware interaction.
    """
    def __init__(self, camera, num_images=10, sleep_time=0.1, 
                 perform_analysis=False, visualizer=None, gps_handler=None):
        """
        Initialize an exposure sequence controller.
        
        Args:
            camera: CameraAcquisition instance for hardware control
            num_images: Number of images to capture in sequence
            sleep_time: Time to sleep between exposures (seconds)
            perform_analysis: Whether to analyze images for sun position
            visualizer: HeadingVisualizer instance for displaying heading
            gps_handler: GPSHandler instance for GPS data
        """
        self.camera = camera
        self.num_images = num_images
        self.sleep_time = sleep_time
        self.perform_analysis = perform_analysis
        self.visualizer = visualizer
        self.gps_handler = gps_handler
        
        # Analysis state variables
        self.sunLocation = None
        self.deltaAlt = None
        self.deltaAzi = None
        self.edges = None
        self.sunAzi = None
        self.is_running = False
        
    def run(self, shared_state=None):
        """
        Run the exposure sequence, capturing and processing images.
        
        Args:
            shared_state: Optional SharedState object for signaling between threads
        """
        logger.info(f"Starting exposure sequence: {self.num_images} images")
        self.is_running = True
        is_first_image = True

        
        
        try:
            # Initialize camera if not already done
            if not self.camera.initialize():
                logger.error("Camera initialization failed")
                return False
                
            # Start acquisition
            if not self.camera.start_acquisition():
                logger.error("Failed to start acquisition")
                return False
            
            # ADDED: If we're doing analysis, wait for first GPS fix from GPSHandler
            if self.perform_analysis and self.gps_handler:
                logger.info("Waiting for initial GPS fix...")
                gps_location = self.gps_handler.wait_for_fix(timeout=10)
                if gps_location:
                    lat, lon = gps_location
                    logger.info(f"Got initial GPS fix: Lat={lat:.6f}, Lon={lon:.6f}")
                    
                    # Set initial coordinates in camera for metadata
                    with self.camera.state_lock:
                        self.camera.init_latitute = lat
                        self.camera.init_longitude = lon
                else:
                    # Use default coordinates from GPS handler if available
                    if self.gps_handler.default_lat is not None and self.gps_handler.default_lon is not None:
                        lat, lon = self.gps_handler.default_lat, self.gps_handler.default_lon
                        logger.info(f"Using default coordinates: Lat={lat}, Lon={lon}")
                        with self.camera.state_lock:
                            self.camera.init_latitute = lat
                            self.camera.init_longitude = lon
                    else:
                        logger.warning("Could not get GPS fix within timeout. Analysis may be limited.")
            
            # Main exposure loop
            for i in range(self.num_images):
                # Check if we should stop (from external signal)
                if shared_state and not shared_state.is_running:
                    logger.info("Stopping exposure sequence due to external signal")
                    break
                    
                logger.info(f"--- Acquiring image {i + 1}/{self.num_images} ---")
                
                # Acquire single frame
                image_data, buffer = self.camera.acquire_image()
                
                # Handle acquisition failure
                if image_data is None or buffer is None:
                    logger.warning(f"Failed to acquire image {i + 1}. Skipping.")
                    time.sleep(0.5)  # Brief pause after failure
                    continue
                    
                # Process the image if requested
                if self.perform_analysis:
                    self._process_image(image_data, i, is_first_image)
                    # First image is now processed
                    if is_first_image and self.edges is not None:
                        is_first_image = False
                        
                # Save the image
                self.camera.save_fits(image_data, i + 1)
                
                # Save JPEG with optional edge overlay
                self.camera.save_jpeg(image_data, i + 1, self.edges,self.sunLocation)

                # Queue buffer back
                self.camera.queue_buffer(buffer)
                
                # Sleep between frames
                if self.sleep_time > 0 and i < self.num_images - 1:
                    time.sleep(self.sleep_time)
                    
            return True
            
        except Exception as e:
            logger.error(f"Error in exposure sequence: {e}", exc_info=True)
            return False
        finally:
            self.is_running = False
            # Camera cleanup is handled by the caller
            
    def _process_image(self, image_data, image_index, is_first_image):
        """Process an image for sun detection and heading calculation."""
        try:
            processor = ImageProcessor(image_data.astype('float'))
            local_time = datetime.now()
            logger.info(f"Processing image {image_index + 1} at Local Time: {local_time}")
            
            # Thread-safe access to GPS coordinates
            with self.camera.state_lock:
                current_lat = self.camera.init_latitute
                current_lon = self.camera.init_longitude
                
            # First image processing
            if is_first_image:
                if current_lat is None or current_lon is None:
                    logger.warning("GPS coordinates not available for first image analysis")
                    return
                    
                # Calculate sun position based on GPS and time
                processor.calculateSun(current_lat, current_lon, local_time)
                self.sunAzi = processor.sunAzi
                logger.info(f"Calculated Sun Position: Alt={processor.sunAlt:.2f}, Azi={processor.sunAzi:.2f}")
                
                # Detect sun and horizon in image
                processor.sunDetectionSEP(display=False)
                detected_edges = processor.edgeDetection(display=False)
                
                logger.info(f"Detected edges: {detected_edges}")

                if processor.sunLocation is not None and detected_edges is not None:
                    self.edges = detected_edges
                    sun_x, sun_y, _ = processor.sunLocation
                    self.sunLocation = (sun_x, sun_y)
                    allsky_x, allsky_y, allsky_r = detected_edges[0,0], detected_edges[0,1], detected_edges[0,2]
                    
                    # Calculate measured sun position
                    processor.calSunAltAzi((allsky_x, allsky_y), (sun_x, sun_y), allsky_r)
                    logger.info(f"Measured Sun Position: Alt={processor.sunMeasuredAlt:.2f}, Azi={processor.sunMeasuredAzi:.2f}")
                    
                    # Calculate initial differences
                    self.deltaAlt = processor.sunAlt - processor.sunMeasuredAlt
                    self.deltaAzi = processor.sunAzi - processor.sunMeasuredAzi
                    
                    # Thread-safe update of camera state
                    with self.camera.state_lock:
                        self.camera.measured_alt = processor.sunMeasuredAlt
                        self.camera.measured_azi = processor.sunMeasuredAzi
                        self.camera.head_diff = self.deltaAzi
                    
                    logger.info(f"Initial Delta Alt: {self.deltaAlt:.2f}, Delta Azi: {self.deltaAzi:.2f}")
                    logger.info(f"Initial Heading difference = {self.deltaAzi:.2f}")
                    
                    # Update visualizer if available
                    if self.visualizer:
                        self.visualizer.update_heading(self.deltaAzi)
                else:
                    logger.warning("Sun or horizon detection failed on first image")
            
            # Subsequent images processing
            elif self.edges is not None and self.sunAzi is not None:
                processor.sunDetectionSEP(display=False)
                if processor.sunLocation is not None:
                    sun_x, sun_y, _ = processor.sunLocation
                    allsky_x, allsky_y, allsky_r = self.edges[0,0], self.edges[0,1], self.edges[0,2]
                    self.sunLocation = (sun_x, sun_y)
                    processor.calSunAltAzi((allsky_x, allsky_y), (sun_x, sun_y), allsky_r)
                    logger.info(f"Measured Sun Position: Alt={processor.sunMeasuredAlt:.2f}, Azi={processor.sunMeasuredAzi:.2f}")
                    
                    head_diff = self.sunAzi - processor.sunMeasuredAzi
                    
                    # Thread-safe update of camera state
                    with self.camera.state_lock:
                        self.camera.measured_alt = processor.sunMeasuredAlt
                        self.camera.measured_azi = processor.sunMeasuredAzi
                        self.camera.head_diff = head_diff
                    
                    logger.info(f"Heading difference = {head_diff:.2f}")
                    
                    # Update visualizer
                    if self.visualizer:
                        self.visualizer.update_heading(head_diff)
                    
                    # Calculate and update latitude/longitude
                    latitude, longitude = processor.calculateLatLon(processor.sunMeasuredAlt, processor.sunMeasuredAzi, local_time)
                    if latitude is not None and longitude is not None:
                        with self.camera.state_lock:
                            self.camera.measured_lat = latitude
                            self.camera.measured_lon = longitude
                        
                        logger.info(f"Calculated Lat/Lon: {latitude:.6f}, {longitude:.6f}")
                        
                        # Compare with current GPS
                        if current_lat is not None and current_lon is not None:
                            lat_diff = latitude - current_lat
                            lon_diff = longitude - current_lon
                            logger.info(f"Difference from GPS: dLat={lat_diff:.6f}, dLon={lon_diff:.6f}")
                else:
                    logger.warning(f"Sun detection failed for image {image_index + 1}")
            
        except Exception as e:
            logger.error(f"Error processing image {image_index + 1}: {e}", exc_info=True)


# --- Shared State for Thread Communication ---
class SharedState:
    """Shared state object for communication between threads."""
    def __init__(self):
        self.is_running = True  # Flag to signal threads to stop
        self.acquisition_done = threading.Event()  # Signal when acquisition is complete
        self.tk_ready_to_close = threading.Event()  # Signal when Tkinter UI is ready to close


# --- Argument Parsing ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="IDS Peak Camera Acquisition Script")
    parser.add_argument("--exposure", type=float, default=0.2, help="Exposure time in milliseconds")
    parser.add_argument("--images", type=int, default=5, help="Number of images to acquire")
    parser.add_argument("--sleep", type=float, default=0.1, help="Time in seconds between exposures")
    parser.add_argument("--buffers", type=int, default=None, help="Number of buffers to allocate (default: minimum required)")
    parser.add_argument("--output", type=str, default="output", help="Directory to save FITS files")
    parser.add_argument("--perform_analysis", action='store_true', help="Set this flag to perform sun analysis on the images")
    parser.add_argument("--gps_update_frequency", type=float, default=2, help="Frequency of GPS updates in seconds")
    return parser.parse_args()


# --- Main Execution Logic ---
def main():
    args = parse_args()
    state = SharedState()

    # Initialize Tkinter and HeadingVisualizer in the main thread
    root = tk.Tk()
    visualizer = HeadingVisualizer(root)
    
    # Create and start GPS handler (runs at specified frequency, logs to system)
    gps_handler = GPSHandler(
        update_frequency_hz=args.gps_update_frequency,
        default_lat=args.default_lat,
        default_lon=args.default_lon,
        log_to_system=args.log_gps_updates
    )
    gps_handler.start()

    # Create CameraAcquisition instance (handles hardware)
    camera = CameraAcquisition(
        exposure_time_ms=args.exposure,
        num_buffers=args.buffers,
        output_dir=args.output
    )
    
    # Create ExposureSequence instance (handles acquisition loop)
    sequence = ExposureSequence(
        camera=camera,
        num_images=args.images,
        sleep_time=args.sleep,
        perform_analysis=args.perform_analysis,
        visualizer=visualizer,
        gps_handler=gps_handler  # Pass GPS handler to sequence
    )

    # --- Run Acquisition in Thread ---
    def run_acquisition_thread():
        try:
            sequence.run(state)
        except Exception as e:
            logger.error(f"Error in acquisition thread: {e}", exc_info=True)
        finally:
            state.acquisition_done.set()
            # Schedule GUI cleanup if main thread still running
            if state.is_running and root and root.winfo_exists():
                root.after(100, prepare_shutdown)

    # --- Shutdown Functions ---
    def prepare_shutdown():
        """First stage of shutdown - signal threads to stop."""
        if not state.is_running:  # Avoid running multiple times
            return
        logger.info("Shutdown requested. Preparing...")
        state.is_running = False  # Signal acquisition to stop

        # Schedule next phase when Tkinter is idle
        if root and root.winfo_exists():
            root.after_idle(finish_shutdown)
        else:
            state.tk_ready_to_close.set()

    def finish_shutdown():
        """Second stage of shutdown - destroy Tkinter resources."""
        logger.info("Finishing shutdown: Destroying Tkinter window...")
        if root and root.winfo_exists():
            try:
                root.destroy()
            except tk.TclError as e:
                logger.warning(f"Error destroying Tkinter window: {e}")
        state.tk_ready_to_close.set()
        logger.info("Tkinter window destroyed.")

    # --- Start Application ---
    # Handle window close events
    root.protocol("WM_DELETE_WINDOW", prepare_shutdown)

    # Start acquisition thread
    acquisition_thread = threading.Thread(
        target=run_acquisition_thread,
        daemon=True
    )
    acquisition_thread.start()

    try:
        # Run Tkinter main loop in main thread
        logger.info("Starting Tkinter main loop...")
        root.mainloop()
        logger.info("Tkinter main loop ended.")
    except Exception as e:
        logger.error(f"Error in Tkinter main loop: {e}", exc_info=True)
    finally:
        # Ensure proper shutdown
        if state.is_running:
            prepare_shutdown()

        # Wait for acquisition thread to complete
        if acquisition_thread.is_alive():
            logger.info("Waiting for acquisition thread to complete...")
            signaled = state.acquisition_done.wait(timeout=10)
            if not signaled:
                logger.warning("Acquisition thread did not complete within timeout")

        # Wait for Tkinter cleanup
        if not state.tk_ready_to_close.is_set():
            logger.info("Waiting for Tkinter resources to clean up...")
            state.tk_ready_to_close.wait(timeout=5)

        # Stop GPS handler
        if gps_handler:
            logger.info("Stopping GPS handler...")
            gps_handler.stop()

        # Final camera cleanup
        logger.info("Performing final camera cleanup...")
        camera.cleanup()

        # Help garbage collection
        camera = None
        sequence = None
        visualizer = None
        gps_handler = None
        root = None
        gc.collect()

        logger.info("Application exited.")


if __name__ == "__main__":
    main()
