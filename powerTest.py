import numpy as np
import time
from detectSun import ImageProcessor
import glob
import os
from astropy.io import fits
from idsExposure import ExposureSequence, CameraAcquisition


if __name__ == "__main__":
    fits_dir = "output/fits"
    camera = CameraAcquisition()

    # Initialize and open the camera ONCE
    if not camera.initialize():
        print("Camera initialization failed")
        exit(1)

    run_seconds = 3 * 60 * 60  # 3 hours in seconds
    start_time = time.time()

    try:
        while True:
            # Stop after 3 hours
            if time.time() - start_time > run_seconds:
                print("Reached 3 hours. Stopping acquisition loop.")
                break

            # Take a new exposure and save it
            exposure = ExposureSequence(camera, num_images=1, save_images=False)
            exposure.run()

            # Find the latest FITS file
            fits_files = sorted(
                glob.glob(os.path.join(fits_dir, "*.fits")),
                key=os.path.getmtime,
                reverse=True
            )
            if not fits_files:
                print("No FITS files found in output/fits")
                break

            latest_fits = fits_files[0]
            print(f"Analyzing latest FITS file: {latest_fits}")

            try:
                with fits.open(latest_fits) as hdul:
                    img = hdul[0].data
                    if img is None:
                        print(f"Warning: No image data in {latest_fits}")
                        continue
                    processor = ImageProcessor(img.astype('float'))
                    start = time.time()
                    processor.sunDetectionSEP(display=False)
                    processor.edgeDetection(display=False)
                    end = time.time()
                    print(f"{os.path.basename(latest_fits)}: Analysis loop in {end - start:.2f} seconds")
            except Exception as e:
                print(f"Error processing {latest_fits}: {e}")

            # Wait before next exposure (adjust as needed)
            time.sleep(2)

    except KeyboardInterrupt:
        print("Stopping acquisition loop.")

    finally:
        camera.cleanup()
        print("Camera cleanup completed.")