import argparse
import logging
from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension
import numpy as np
from astropy.io import fits
import os
from datetime import datetime
from detectSun import ImageProcessor

# Get the program name
program_name = os.path.basename(__file__)
# Create a logger for the program
logging.getLogger('idsExposure')
logging.basicConfig(level=logging.INFO, 
                    format=f"%(asctime)s.%(msecs)03d %(levelname)s {program_name}:%(lineno)s %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S")

class CameraAcquisition:
    def __init__(self, exposure_time_ms=0.02, num_images=1, num_buffers=None, output_dir="output", perform_analysis=False):
        self.exposure_time_ms = exposure_time_ms
        self.num_images = num_images
        self.num_buffers = num_buffers
        self.output_dir = output_dir
        self.device = None
        self.data_stream = None
        self.remote_nodemap = None
        self.perform_analysis = perform_analysis

    def setup_device(self):
        """Initializes the library and sets up the device manager."""
        ids_peak.Library.Initialize()
        device_manager = ids_peak.DeviceManager.Instance()
        device_found_callback = device_manager.DeviceFoundCallback(
            lambda found_device: logging.info(f"Found device: Key={found_device.Key()}"))

        # Register device found callback
        device_manager.RegisterDeviceFoundCallback(device_found_callback)

        # Update the device manager to detect devices
        device_manager.Update()

        if device_manager.Devices().empty():
            logging.error("No device found. Exiting program.")
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
            logging.info(f"Exposure time set to {self.exposure_time_ms} ms")
        except Exception as e:
            logging.error(f"Error setting exposure time: {e}")

        # Set the pixel format
        self.remote_nodemap.FindNode("PixelFormat").SetCurrentEntry('Mono12')
        logging.info("Pixel format set to Mono12")
        self.remote_nodemap.FindNode("TLParamsLocked").SetValue(1)
        logging.info("Transport layer parameters locked")

    def allocate_buffers(self):
        """Allocates buffers for the data stream."""
        self.data_stream = self.device.DataStreams()[0].OpenDataStream()
        payload_size = self.remote_nodemap.FindNode("PayloadSize").Value()
        buffer_count = self.num_buffers or self.data_stream.NumBuffersAnnouncedMinRequired()

        for _ in range(buffer_count):
            buffer = self.data_stream.AllocAndAnnounceBuffer(payload_size)
            self.data_stream.QueueBuffer(buffer)

        logging.info(f"Allocated and queued {buffer_count} buffers.")

    def save_fits(self, image_data, exposure_num):
        """
        Save the given image data to a FITS file with a specific naming convention.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H_%M_%S")
        run_number = f"{exposure_num:02d}"
        filename = f"image_{date_str}_{time_str}_{run_number}.fits"
        filepath = os.path.join(self.output_dir, filename)

        hdu = fits.PrimaryHDU(image_data)
        hdu.header['EXPTIME'] = (self.exposure_time_ms, 'Exposure time in milliseconds')
        hdu.writeto(filepath, overwrite=True)
        logging.info(f"Saved FITS file: {filepath}")

    def process_images(self):
        """Processes a configurable number of images."""
        logging.info("Starting image acquisition...")
        self.data_stream.StartAcquisition()
        self.remote_nodemap.FindNode("AcquisitionStart").Execute()

        for i in range(self.num_images):
            try:
                buffer = self.data_stream.WaitForFinishedBuffer(1000)
                img = ids_peak_ipl_extension.BufferToImage(buffer)
                image_data = img.get_numpy_2D_16()
                
                if i == 0:
                    processor = ImageProcessor(image_data.astype('float')) 
                    logging.info(f"Image shape: {image_data.shape}")
                    processor.initial_latitude = 24.874241
                    processor.initial_longitude = 120.947295
                    localTime = datetime.now()
                    logging.info(f"Local Time: {localTime}")

                    if self.perform_analysis:
                        processor.calculateSun(processor.initial_latitude, processor.initial_longitude, localTime)
                        processor.sunDetectionSEP(display=True)
                        edges = processor.edgeDetection(display=False)
                        sun_x, sun_y, sun_r = processor.sunLocation
                        allsky_x, allsky_y, allsky_r = edges[0,0], edges[0,1], edges[0,2]
                        
                        logging.info(f"{processor.calSunAltAzi((allsky_x, allsky_y), (sun_x, sun_y), allsky_r)}")
                        
                        deltaAlt = processor.sunAlt - processor.sunMeasuredAlt
                        deltaAzi = processor.sunAzi - processor.sunMeasuredAzi
                        logging.info(f"Delta Altitude: {deltaAlt} Delta Azimuth: {deltaAzi}")

                else:
                    processor = ImageProcessor(image_data.astype('float'))
                    if self.perform_analysis:
                        processor.sunDetectionSEP()
                        edges = processor.edgeDetection()
                        localTime = datetime.now()
                        logging.info(f"Local Time: {localTime}")

                        sun_x, sun_y, sun_r = processor.sunLocation
                        processor.edge = edges
                        allsky_x, allsky_y, allsky_r = edges[0,0], edges[0,1], edges[0,2]
                        logging.info(f"Sun Location: {processor.sunLocation}")
                        logging.info(f"Horizon: {edges}")
                        logging.info(f"{processor.calSunAltAzi((allsky_x, allsky_y), (sun_x, sun_y), allsky_r)}")
                        
                        Alt = processor.sunMeasuredAlt + deltaAlt
                        Azi = processor.sunMeasuredAzi + deltaAzi
                        
                        logging.info(f"Altitude: {Alt} Azimuth: {Azi}")
                        latitude, longitude = processor.calculateLatLon(Alt, Azi, localTime)
                        logging.info(f"Calculated Latitude: {latitude} Longitude: {longitude}")
                                    
                self.save_fits(image_data, i)
                self.data_stream.QueueBuffer(buffer)
                logging.info(f"Processed image {i + 1}/{self.num_images}")
            except Exception as e:
                logging.error(f"Error processing image {i + 1}: {e}")

        self.remote_nodemap.FindNode("AcquisitionStop").Execute()
        self.data_stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)

    def cleanup(self):
        """Cleans up resources after acquisition."""
        logging.info("Cleaning up resources...")
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
            logging.error(f"Error during cleanup: {e}")

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
            logging.error(f"Exception during acquisition: {e}")
        finally:
            self.cleanup()


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="IDS Peak Camera Acquisition Script")
    parser.add_argument("--exposure", type=float, default=0.02, help="Exposure time in milliseconds")
    parser.add_argument("--images", type=int, default=2, help="Number of images to acquire")
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
        perform_analysis=args.perform_analysis
    )
    acquisition.run()


if __name__ == "__main__":
    main()
