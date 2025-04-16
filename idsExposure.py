from logger_config import setup_logging
setup_logging()

import argparse
import logging
from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension
import numpy as np
from astropy.io import fits
import os
from datetime import datetime
from detectSun import ImageProcessor
from ubloxReader import GPSReader
import time
import threading
import math
import tkinter as tk

program_name = os.path.basename(__file__)
logger = logging.getLogger(program_name)

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
        self.heading = heading
        self._update_arrow()
        self._update_heading_text()

    def _update_arrow(self):
        """Update the arrow direction on the canvas."""
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
        self.canvas.itemconfig(self.heading_text, text=f"Heading: {self.heading:.1f}°")

class CameraAcquisition:
    def __init__(self, exposure_time_ms=0.02, num_images=1, num_buffers=None, 
                 output_dir="output", sleep_time=0, perform_analysis=False, 
                 gps_update_frequency=1, visualizer=None):
        
        self.exposure_time_ms = exposure_time_ms
        self.num_images = num_images
        self.num_buffers = num_buffers
        self.output_dir = output_dir
        self.sleep_time = sleep_time
        self.perform_analysis = perform_analysis
        self.gps_update_frequency = gps_update_frequency

        self.device = None
        self.data_stream = None
        self.remote_nodemap = None

        self.init_latitute = None
        self.init_longitude = None

        self.measured_alt = None
        self.measured_azi = None

        self.measured_lat = None
        self.measured_lon = None

        self.head_diff = None
        self.gpsHeadingDiff = None

        self.gps_thread = None
        self.gps_running = False
        self.gps_records = []

        # Reference to the HeadingVisualizer instance
        self.visualizer = visualizer

    def start_gps_thread(self):
        """Starts a thread to update GPS location at a specified frequency."""
        self.gps_running = True
        self.gps_thread = threading.Thread(target=self.update_gps_location)
        self.gps_thread.start()

    def stop_gps_thread(self):
        """Stops the GPS update thread."""
        self.gps_running = False
        if self.gps_thread:
            self.gps_thread.join()

    def update_gps_location(self):
        """Continuously updates the GPS location."""
        gps = GPSReader(system='GNSS').connect()
        while self.gps_running:
            coords = gps.get_coordinates()
            if coords:
                self.init_latitute, self.init_longitude, gps_time, sats = coords
                logger.info(f"Updated GPS location: Latitude={self.init_latitute}, Longitude={self.init_longitude}, Time={gps_time}, Satellites={sats}")
                self.gps_records.append((self.init_latitute, self.init_longitude))
                if len(self.gps_records) == 3:
                    self.calculate_gps_heading_diff()
            time.sleep(self.gps_update_frequency)
        gps.disconnect()

    def calculate_gps_heading_diff(self):
        """Calculates the heading difference based on the first three GPS records."""
        if len(self.gps_records) < 3:
            return

        lat1, lon1 = self.gps_records[0]
        lat2, lon2 = self.gps_records[1]
        lat3, lon3 = self.gps_records[2]

        vector1 = (lon2 - lon1, lat2 - lat1)
        vector2 = (lon3 - lon2, lat3 - lat2)
        logger.info(f"vec1 {vector1} {vector2}")
        angle1 = math.atan2(vector1[1], vector1[0])
        angle2 = math.atan2(vector2[1], vector2[0])

        angle1_deg = math.degrees(angle1) % 360
        angle2_deg = math.degrees(angle2) % 360
        logger.info(f"angles {angle1_deg}  {angle2_deg}")
        self.gpsHeadingDiff = (angle2_deg + angle1_deg) / 2
        logger.info(f"Calculated GPS heading difference: {self.gpsHeadingDiff} degrees")

    def setup_device(self):
        """Initializes the library and sets up the device manager."""
        ids_peak.Library.Initialize()
        device_manager = ids_peak.DeviceManager.Instance()
        device_found_callback = device_manager.DeviceFoundCallback(
            lambda found_device: logger.info(f"Found device: Key={found_device.Key()}"))

        device_manager.RegisterDeviceFoundCallback(device_found_callback)
        device_manager.Update()

        if device_manager.Devices().empty():
            logger.error("No device found. Exiting program.")
            return False

        self.device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        return True

    def configure_device(self):
        """Configures the device settings."""
        self.remote_nodemap = self.device.RemoteDevice().NodeMaps()[0]
        self.remote_nodemap.FindNode("UserSetSelector").SetCurrentEntry("Default")
        self.remote_nodemap.FindNode("UserSetLoad").Execute()
        self.remote_nodemap.FindNode("UserSetLoad").WaitUntilDone()

        try:
            exposure_time_node = self.remote_nodemap.FindNode("ExposureTime")
            exposure_time_node.SetValue(self.exposure_time_ms * 1000)
            logger.info(f"Exposure time set to {self.exposure_time_ms} ms")
        except Exception as e:
            logger.error(f"Error setting exposure time: {e}")

        self.remote_nodemap.FindNode("PixelFormat").SetCurrentEntry('Mono12')
        logger.info("Pixel format set to Mono12")
        self.remote_nodemap.FindNode("TLParamsLocked").SetValue(1)
        logger.info("Transport layer parameters locked")

    def allocate_buffers(self):
        """Allocates buffers for the data stream."""
        self.data_stream = self.device.DataStreams()[0].OpenDataStream()
        payload_size = self.remote_nodemap.FindNode("PayloadSize").Value()
        buffer_count = self.num_buffers or self.data_stream.NumBuffersAnnouncedMinRequired()

        for _ in range(buffer_count):
            buffer = self.data_stream.AllocAndAnnounceBuffer(payload_size)
            self.data_stream.QueueBuffer(buffer)

        logger.info(f"Allocated and queued {buffer_count} buffers.")

    def save_fits(self, image_data, exposure_num):
        """Save the image data to a FITS file with metadata."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            now = datetime.now()
            date_str = now.strftime("%Y%m%d")
            time_str = now.strftime("%H_%M_%S")
            run_number = f"{exposure_num:02d}"
            filename = f"image_{date_str}_{time_str}_{run_number}.fits"
            filepath = os.path.join(self.output_dir, filename)

            hdu = fits.PrimaryHDU(image_data)
            hdu.header['EXPTIME'] = (self.exposure_time_ms, 'Exposure time in milliseconds')
            
            if self.init_latitute is not None:
                hdu.header['INIT_LAT'] = (self.init_latitute, 'Initial latitude from GPS')
            if self.init_longitude is not None:
                hdu.header['INIT_LON'] = (self.init_longitude, 'Initial longitude from GPS')
            if self.measured_lat is not None:
                hdu.header['MEAS_LAT'] = (self.measured_lat, 'Measured Latitude')
            if self.measured_lon is not None:
                hdu.header['MEAS_LON'] = (self.measured_lon, 'Measured Longitude')
            if self.measured_alt is not None:
                hdu.header['MEAS_ALT'] = (self.measured_alt, 'Measured Alt')
            if self.measured_azi is not None:
                hdu.header['MEAS_AZI'] = (self.measured_azi, 'Measured Azi')
            if self.head_diff is not None:
                hdu.header['HEAD_DIF'] = (self.head_diff, 'Camera heading')
            if self.gpsHeadingDiff is not None:
                hdu.header['GPS_HEAD'] = (self.gpsHeadingDiff, 'GPS heading difference')

            hdu.writeto(filepath, overwrite=True)
            logger.info(f"Saved FITS file: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save FITS file: {filepath}, error: {e}")
            raise

    def process_images(self):
        """Processes a configurable number of images."""
        logger.info("Starting image acquisition...")
        self.data_stream.StartAcquisition()
        self.remote_nodemap.FindNode("AcquisitionStart").Execute()

        self.start_gps_thread()

        for i in range(self.num_images):
            try:
                buffer = self.data_stream.WaitForFinishedBuffer(1000)
                img = ids_peak_ipl_extension.BufferToImage(buffer)
                image_data = img.get_numpy_2D_16()
                
                processor = ImageProcessor(image_data.astype('float'))
                
                if i == 0:
                    try:
                        self.init_latitute, self.init_longitude = processor.getInitialLocationFromGPS()
                    except:
                        logger.error("Failed to get initial location from GPS")
                        self.init_latitute, self.init_longitude = (24.874241, 120.947295)
                    logger.info(f" Lat and Lon from GPS {self.init_latitute} {self.init_longitude}")
                    logger.info(f"Image shape: {image_data.shape}")
                    localTime = datetime.now()
                    logger.info(f"Local Time: {localTime}")

                    if self.perform_analysis:
                        processor.calculateSun(self.init_latitute, self.init_longitude, localTime)
                        processor.sunDetectionSEP(display=False)
                        edges = processor.edgeDetection(display=False)
                        sun_x, sun_y, sun_r = processor.sunLocation
                        allsky_x, allsky_y, allsky_r = edges[0,0], edges[0,1], edges[0,2]
                        
                        logger.info(f"{processor.calSunAltAzi((allsky_x, allsky_y), (sun_x, sun_y), allsky_r)}")
                        sunAlt = processor.sunAlt
                        sunAzi = processor.sunAzi
                        deltaAlt = processor.sunAlt - processor.sunMeasuredAlt
                        deltaAzi = processor.sunAzi - processor.sunMeasuredAzi

                        logger.info(f"Measured Sun alt, azi = {processor.sunMeasuredAlt} {processor.sunMeasuredAzi}")
                        logger.info(f"Delta Altitude: {deltaAlt} Delta Azimuth: {deltaAzi}")

                        self.head_diff = sunAzi - processor.sunMeasuredAzi
                        logger.info(f"Heading difference (camera heading): {self.head_diff}")
                        # Update the heading visualizer if it exists
                        if self.visualizer:
                            self.visualizer.root.after(0, self.visualizer.update_heading, self.head_diff)

                else:
                    if self.perform_analysis:
                        processor.sunDetectionSEP()
                        localTime = datetime.now()
                        logger.info(f"Local Time: {localTime}")

                        sun_x, sun_y, sun_r = processor.sunLocation
                        allsky_x, allsky_y, allsky_r = edges[0,0], edges[0,1], edges[0,2]
                        logger.info(f"Sun Location: {processor.sunLocation}")
                        logger.info(f"Horizon: {edges}")
                        
                        processor.calSunAltAzi((allsky_x, allsky_y), (sun_x, sun_y), allsky_r)
                        
                        logger.info(f"Measured Sun alt, azi = {processor.sunMeasuredAlt} {processor.sunMeasuredAzi}")
                        
                        Alt = processor.sunMeasuredAlt
                        Azi = processor.sunMeasuredAzi
                        
                        self.measured_alt = Alt
                        self.measured_azi = Azi
                        
                        logger.info(f"Measured Altitude: {Alt} Azimuth: {Azi}")

                        self.head_diff = sunAzi - Azi 
                        logger.info(f"Heading difference = {self.head_diff}")
                        # Update the heading visualizer if it exists
                        if self.visualizer:
                            self.visualizer.root.after(0, self.visualizer.update_heading, self.head_diff)

                        latitude, longitude = processor.calculateLatLon(Alt, Azi, localTime)
                        logger.info(f"Calculated Latitude: {latitude} Longitude: {longitude}")
                        self.measured_lat = latitude
                        self.measured_lon = longitude
                        try:
                            gps = GPSReader(system='GNSS').connect()
                            coords = gps.get_coordinates()
                            if coords:
                                lat, lon, gps_time, sats = coords
                            else:
                                lat = -999.0
                                lon = -999.0
                                sats = 0
                        finally:
                            gps.disconnect()

                        logger.info(f"Location difference {self.measured_lat - lat} {self.measured_lon - lon} {sats}")    
                
                self.save_fits(image_data, i)
                self.data_stream.QueueBuffer(buffer)
                logger.info(f"Processed image {i + 1}/{self.num_images}")

                logger.info(f"Sleep for {self.sleep_time} seconds.")
                time.sleep(self.sleep_time)
            except Exception as e:
                logger.error(f"Error processing image {i + 1}: {e}")

        self.remote_nodemap.FindNode("AcquisitionStop").Execute()
        self.data_stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)

        self.stop_gps_thread()

    def cleanup(self):
        """Cleans up resources after acquisition."""
        logger.info("Cleaning up resources...")
        try:
            if self.data_stream and self.data_stream.IsGrabbing():
                self.data_stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
            if self.remote_nodemap:
                self.remote_nodemap.FindNode("AcquisitionStop").Execute()
            self.data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
            for buffer in self.data_stream.AnnouncedBuffers():
                self.data_stream.RevokeBuffer(buffer)
            self.remote_nodemap.FindNode("TLParamsLocked").SetValue(0)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        if self.device:
            ids_peak.Library.Close()

    def run(self):
        """Main method to execute the camera acquisition process."""
        try:
            if not self.setup_device():
                return
            self.configure_device()
            self.allocate_buffers()
            self.process_images()
        except Exception as e:
            logger.error(f"Exception during acquisition: {e}")
        finally:
            self.cleanup()

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="IDS Peak Camera Acquisition Script")
    parser.add_argument("--exposure", type=float, default=0.02, help="Exposure time in milliseconds")
    parser.add_argument("--images", type=int, default=100, help="Number of images to acquire")
    parser.add_argument("--sleep", type=int, default=0, help="time of seconds between exposures")
    parser.add_argument("--buffers", type=int, default=None, help="Number of buffers to allocate")
    parser.add_argument("--output", type=str, default="output", help="Directory to save FITS files")
    parser.add_argument("--perform_analysis", action='store_true', help="Set this flag to perform analysis on the images")
    parser.add_argument("--gps_update_frequency", type=int, default=0.1, help="Frequency of GPS updates in seconds")
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Create a shared variable to track the state across threads
    class SharedState:
        def __init__(self):
            self.is_running = True
            self.acquisition_done = threading.Event()
            self.tk_ready_to_close = threading.Event()
    
    state = SharedState()

    # Initialize Tkinter and HeadingVisualizer
    root = tk.Tk()
    visualizer = HeadingVisualizer(root)

    # Create CameraAcquisition instance with the visualizer
    acquisition = CameraAcquisition(
        exposure_time_ms=args.exposure,
        num_images=args.images,
        num_buffers=args.buffers,
        output_dir=args.output,
        sleep_time=args.sleep,
        perform_analysis=args.perform_analysis,
        gps_update_frequency=args.gps_update_frequency,
        visualizer=visualizer
    )

    # Modified acquisition thread with callback
    def run_acquisition():
        try:
            acquisition.run()
        except Exception as e:
            logger.error(f"Acquisition thread error: {e}")
        finally:
            # Signal that acquisition is done
            state.acquisition_done.set()
            # Schedule GUI cleanup if the main thread is still running
            if state.is_running:
                root.after(100, prepare_shutdown)
    
    # Function to safely shutdown in stages
    def prepare_shutdown():
        """First stage of shutdown - clean up Tkinter references"""
        logger.info("Preparing for shutdown...")
        # Clear any references to Tkinter objects in the acquisition instance
        acquisition.visualizer = None
        
        # Use after_idle to schedule the next phase when Tkinter is ready
        root.after_idle(finish_shutdown)
    
    def finish_shutdown():
        """Second stage of shutdown - destroy Tkinter resources"""
        logger.info("Finishing shutdown...")
        # Set flag indicating Tkinter is ready to close
        state.tk_ready_to_close.set()
        # Destroy the root window
        root.destroy()

    # Add a protocol handler for window close events
    root.protocol("WM_DELETE_WINDOW", prepare_shutdown)
    
    # Start the acquisition thread
    acquisition_thread = threading.Thread(target=run_acquisition)
    acquisition_thread.daemon = True
    acquisition_thread.start()
    
    try:
        # Start the Tkinter main loop
        logger.info("Starting Tkinter main loop")
        root.mainloop()
        logger.info("Tkinter main loop ended")
    except Exception as e:
        logger.error(f"Error in Tkinter main loop: {e}")
    finally:
        # Mark that we're no longer running the main Tkinter loop
        state.is_running = False
        
    # Wait for the acquisition thread to finish if it's still running
    if acquisition_thread.is_alive():
        logger.info("Waiting for acquisition to complete...")
        acquisition_thread.join(timeout=5)
    
    # Wait for Tkinter resources to be properly cleaned up
    if not state.tk_ready_to_close.is_set():
        logger.info("Waiting for Tkinter to clean up resources...")
        # Only wait briefly to avoid hanging
        state.tk_ready_to_close.wait(timeout=2)
        
    # Final cleanup
    try:
        # This helps with releasing any remaining Tkinter resources
        root = None
        visualizer = None
        acquisition = None
    except Exception as e:
        logger.error(f"Error during final cleanup: {e}")

    # Force Python to do some garbage collection
    import gc
    gc.collect()
    
    logger.info("Application exited cleanly")

if __name__ == "__main__":
    main()