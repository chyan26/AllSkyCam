import argparse
import logging
from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension
import numpy as np
from astropy.io import fits
import os
from datetime import datetime


def save_fits(image_data, exposure_num, output_dir="output"):
    """
    Save the given image data to a FITS file with a specific naming convention.
    
    Args:
        image_data: The image data to be saved.
        exposure_num: The exposure number (used for run number).
        output_dir: The directory where the FITS file will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the current date and time
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")  # Format: YYYYMMDD
    time_str = now.strftime("%H_%M_%S")  # Format: HH_MM_SS

    # Create the run number part
    run_number = f"{exposure_num:02d}"  # Format as two digits with leading zeros

    # Construct the filename
    filename = f"image_{date_str}_{time_str}_{run_number}.fits"
    filepath = os.path.join(output_dir, filename)

    # Create and write the FITS file
    hdu = fits.PrimaryHDU(image_data)
    hdu.writeto(filepath, overwrite=True)

    print(f"Saved FITS file: {filepath}")


def setup_device():
    """Initializes the library and sets up the device manager."""
    ids_peak.Library.Initialize()
    device_manager = ids_peak.DeviceManager.Instance()
    device_found_callback = device_manager.DeviceFoundCallback(
        lambda found_device: logging.info(f"Found device: Key={found_device.Key()}"))

    # Register device found callback
    device_found_callback_handle = device_manager.RegisterDeviceFoundCallback(device_found_callback)
    
    # Update the device manager to detect devices
    device_manager.Update()

    if device_manager.Devices().empty():
        logging.error("No device found. Exiting program.")
        return None, None
    
    # Open the first device
    device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
    
    return device, device_found_callback_handle


def configure_device(device, exposure_time_ms=10.0):
    """Configures the device's settings, loads default settings, and prepares the camera."""
    remote_nodemap = device.RemoteDevice().NodeMaps()[0]
    
    remote_nodemap.FindNode("UserSetSelector").SetCurrentEntry("Default")
    remote_nodemap.FindNode("UserSetLoad").Execute()
    remote_nodemap.FindNode("UserSetLoad").WaitUntilDone()
    
    # Set the exposure time (in microseconds, so multiply by 1000 for milliseconds)
    try:
        exposure_time_node = remote_nodemap.FindNode("ExposureTime")
        exposure_time_node.SetValue(exposure_time_ms * 1000)  # Convert ms to µs
        logging.info(f"Exposure time set to {exposure_time_ms} ms")
    except Exception as e:
        logging.error(f"Error setting exposure time: {e}")
    

     # Set the pixel format to 12-bit
    pixel_format_node = remote_nodemap.FindNode("PixelFormat")
    pixel_format_node.SetCurrentEntry('Mono10')
    
    remote_nodemap.FindNode("TLParamsLocked").SetValue(1)
    
    return remote_nodemap


def allocate_buffers(device, num_buffers=None):
    """Allocates buffers for the data stream with optional dynamic allocation."""
    data_stream = device.DataStreams()[0].OpenDataStream()
    payload_size = device.RemoteDevice().NodeMaps()[0].FindNode("PayloadSize").Value()
    buffer_count = num_buffers or data_stream.NumBuffersAnnouncedMinRequired()

    for _ in range(buffer_count):
        buffer = data_stream.AllocAndAnnounceBuffer(payload_size)
        data_stream.QueueBuffer(buffer)

    logging.info(f"Allocated and queued {buffer_count} buffers.")
    return data_stream


def process_images(data_stream, remote_nodemap, image_processor=None, num_images=100):
    """Processes a configurable number of images from the camera."""
    logging.info("Starting image acquisition...")
    
    data_stream.StartAcquisition()
    remote_nodemap.FindNode("AcquisitionStart").Execute()

    for i in range(num_images):
        try:
            buffer = data_stream.WaitForFinishedBuffer(1000)
            img = ids_peak_ipl_extension.BufferToImage(buffer)
            logging.info(f'image buffer size = {img.Width()} {img.Height()}')
            image_data=img.get_numpy_2D_16()
            save_fits(image_data, i, './')

            logging.info(f"Print image data: {image_data.shape}")
            if image_processor:
                image_processor(img, i)  # Call custom processing function

            data_stream.QueueBuffer(buffer)  # Return the buffer to the pool
            logging.info(f"Processed image {i + 1}/{num_images}")

        except Exception as e:
            logging.error(f"Error processing image {i + 1}: {e}")

    remote_nodemap.FindNode("AcquisitionStop").Execute()
    data_stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)


def cleanup(data_stream, remote_nodemap):
    """Stops the acquisition and performs necessary cleanup."""
    logging.info("Cleaning up resources...")
    try:
        # Check if acquisition is running before attempting to stop
        if data_stream.IsGrabbing():
            logging.info("Stopping acquisition...")
            data_stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)

        remote_nodemap.FindNode("AcquisitionStop").Execute()
        remote_nodemap.FindNode("AcquisitionStop").WaitUntilDone()
    except ids_peak.ids_peak.BadAccessException as e:
        logging.warning(f"Acquisition was not running: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during cleanup: {e}")

    # Flush and release buffers
    try:
        data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
        for buffer in data_stream.AnnouncedBuffers():
            data_stream.RevokeBuffer(buffer)
    except Exception as e:
        logging.error(f"Error flushing or revoking buffers: {e}")

    # Unlock parameters
    try:
        remote_nodemap.FindNode("TLParamsLocked").SetValue(0)
    except Exception as e:
        logging.error(f"Error unlocking parameters: {e}")

    try:
        data_stream.KillWait()  # Interrupt any waiting threads
    except Exception as e:
        logging.warning(f"Error killing wait threads: {e}")
  # Interrupt any waiting threads


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="IDS Peak Camera Acquisition Script")
    parser.add_argument("--exposure", type=float, default=10.0, help="Exposure time in milliseconds")
    parser.add_argument("--images", type=int, default=1, help="Number of images to acquire")
    parser.add_argument("--buffers", type=int, default=None, help="Number of buffers to allocate")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    device = None
    data_stream = None
    remote_nodemap = None

    try:
        device, device_found_callback_handle = setup_device()
        if device is None:
            return -1

        remote_nodemap = configure_device(device, args.exposure)
        data_stream = allocate_buffers(device, args.buffers)

        process_images(data_stream, remote_nodemap, num_images=args.images)

    except Exception as e:
        logging.error(f"Exception in the main program: {e}")
        return -2

    finally:
        if device and data_stream and remote_nodemap:
            cleanup(data_stream, remote_nodemap)
            ids_peak.Library.Close()


if __name__ == '__main__':
    main()
