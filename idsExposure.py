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

program_name = os.path.basename(__file__)

logger = logging.getLogger(program_name)


class CameraAcquisition:
    def __init__(self, exposure_time_ms=0.02, num_images=1, num_buffers=None, 
                 output_dir="output", sleep_time = 0, 
                 perform_analysis=False):
        
        self.exposure_time_ms = exposure_time_ms
        self.num_images = num_images
        self.num_buffers = num_buffers
        self.output_dir = output_dir
        self.sleep_time = sleep_time

        self.device = None
        self.data_stream = None
        self.remote_nodemap = None
        self.perform_analysis = perform_analysis

        self.init_latitute = None
        self.init_longitude = None

        self.measured_alt = None
        self.measured_azi = None

        self.measured_lat = None
        self.measured_lon = None

        self.head_diff = None

    def setup_device(self):
        """Initializes the library and sets up the device manager."""
        ids_peak.Library.Initialize()
        device_manager = ids_peak.DeviceManager.Instance()
        device_found_callback = device_manager.DeviceFoundCallback(
            lambda found_device: logger.info(f"Found device: Key={found_device.Key()}"))

        # Register device found callback
        device_manager.RegisterDeviceFoundCallback(device_found_callback)

        # Update the device manager to detect devices
        device_manager.Update()

        if device_manager.Devices().empty():
            logger.error("No device found. Exiting program.")
            return False

        # Open the first device
        self.device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        return True

    def configure_device(self):
        """Configures the device settings."""
        self.remote_nodemap = self.device.RemoteDevice().NodeMaps()[0]
        self.remote_nodemap.FindNode("UserSetSelector").SetCurrentEntry("Default")
        self.remote_nodemap.FindNode("UserSetLoad").Execute()
        self.remote_nodemap.FindNode("UserSetLoad").WaitUntilDone()

        # Set the exposure time
        try:
            exposure_time_node = self.remote_nodemap.FindNode("ExposureTime")
            exposure_time_node.SetValue(self.exposure_time_ms * 1000)  # Convert ms to µs
            logger.info(f"Exposure time set to {self.exposure_time_ms} ms")
        except Exception as e:
            logger.error(f"Error setting exposure time: {e}")

        # Set the pixel format
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
        """
        Save the image data to a FITS file with metadata.
        
        Args:
            image_data: The image data to save
            exposure_num: The exposure sequence number
        
        The file is saved with format: image_YYYYMMDD_HH_MM_SS_NN.fits
        where NN is the zero-padded exposure number
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Generate filename
            now = datetime.now()
            date_str = now.strftime("%Y%m%d")
            time_str = now.strftime("%H_%M_%S")
            run_number = f"{exposure_num:02d}"
            filename = f"image_{date_str}_{time_str}_{run_number}.fits"
            filepath = os.path.join(self.output_dir, filename)

            # Create and populate FITS header
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
                hdu.header['MEAS_AZI'] = (self.measured_azi,'Measured Azi')
            
            if self.head_diff is not None:
                hdu.header['HEAD_DIF'] = (self.head_diff, 'Camera heading')

            # Get location from GPS
            gps = GPSReader(system='GNSS').connect()
            coords = gps.get_coordinates()
            if coords:
                lat, lon, gps_time, sats = coords
                #lat, lon = processor.getInitialLocationFromGPS()
                hdu.header['LATITUDE'] = (lat, 'Latitude from GPS')
                hdu.header['LONGITUD'] = (lon, 'Longitude from GPS')
            gps.disconnect()
            # Write file
            hdu.writeto(filepath, overwrite=True)
            logger.info(f"Saved FITS file: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save FITS file: {filepath}, error: {e}")
            raise

    def save_fits_file(self, image_data, filepath):
        hdu = fits.PrimaryHDU(image_data)
        hdu.header['EXPTIME'] = (self.exposure_time_ms, 'Exposure time in milliseconds')
        
        if self.init_latitute is not None:
            hdu.header['INIT_LAT'] = (self.init_latitute, 'Latitude from GPS')
        if self.init_longitude is not None:
            hdu.header['INIT_LON'] = (self.init_longitude, 'Longitude from GPS')

        try:
            hdu.writeto(filepath, overwrite=True)
            logger.info(f"Saved FITS file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save FITS file: {filepath}, error: {e}")

    def process_images(self):
        """Processes a configurable number of images."""
        logger.info("Starting image acquisition...")
        self.data_stream.StartAcquisition()
        self.remote_nodemap.FindNode("AcquisitionStart").Execute()

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
                        processor.calculateSun(processor.initial_latitude, processor.initial_longitude, localTime)
                        processor.sunDetectionSEP(display=True)
                        edges = processor.edgeDetection(display=True)
                        sun_x, sun_y, sun_r = processor.sunLocation
                        allsky_x, allsky_y, allsky_r = edges[0,0], edges[0,1], edges[0,2]
                        
                        logger.info(f"{processor.calSunAltAzi((allsky_x, allsky_y), (sun_x, sun_y), allsky_r)}")
                        sunAlt = processor.sunAlt
                        sunAzi = processor.sunAzi
                        deltaAlt = processor.sunAlt - processor.sunMeasuredAlt
                        deltaAzi = processor.sunAzi - processor.sunMeasuredAzi

                        logger.info(f"Measured Sun alt, azi = {processor.sunMeasuredAlt} {processor.sunMeasuredAzi}")
                        logger.info(f"Delta Altitude: {deltaAlt} Delta Azimuth: {deltaAzi}")

                        # Calculate heading of the camera
                        self.head_diff = sunAzi - processor.sunMeasuredAzi
                        logger.info(f"Heading difference (camera heading): {self.head_diff}")

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
    parser.add_argument("--images", type=int, default=40, help="Number of images to acquire")
    parser.add_argument("--sleep", type=int, default=0.5, help="time of seconds between exposures")
    parser.add_argument("--buffers", type=int, default=None, help="Number of buffers to allocate")
    parser.add_argument("--output", type=str, default="output", help="Directory to save FITS files")
    parser.add_argument("--perform_analysis", action='store_true', help="Set this flag to perform analysis on the images")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    acquisition = CameraAcquisition(
        exposure_time_ms=args.exposure,
        num_images=args.images,
        num_buffers=args.buffers,
        output_dir=args.output,
        sleep_time=args.sleep,
        perform_analysis=args.perform_analysis
    )
    acquisition.run()


if __name__ == "__main__":
    main()
