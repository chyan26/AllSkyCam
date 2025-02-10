import os
import logging
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV for edge detection
import sep
from datetime import datetime, timedelta
from skyfield.api import Topos, load
from scipy.optimize import minimize
from astropy.io import fits
import glob
from ubloxReader import GPSReader

#program_name = os.path.basename(__file__)

# Create formatters and handlers
#formatter = logging.Formatter(
#    fmt=f"%(asctime)s.%(msecs)03d %(levelname)s {program_name}:%(lineno)s %(message)s",
#    datefmt="%Y-%m-%dT%H:%M:%S"
#)



def showImage(imageData, circle=None, vmax=None):
    plt.figure(figsize=(6,6))
    m, s = np.mean(imageData), np.std(imageData)

    
    plt.imshow(imageData, cmap='gray',origin='lower', vmin=m-s, vmax=m+s)
    if circle is not None:
        for x, y, r in circle[0, :]:
            circle = plt.Circle((x, y), r, color='red', fill=False)
            plt.gca().add_patch(circle)

    plt.axis('off')
    plt.tight_layout()
    plt.show()

class ImageProcessor:
    """
    ImageProcessor is a class for processing images, specifically for detecting the sun in AllSkyCam images.
    Attributes:
        imageFileName (str): The path to the image file.
        image (PIL.Image or None): The loaded image in grayscale.
        image_array (numpy.ndarray or None): The image converted to a NumPy array.
        sun_mask (numpy.ndarray or None): A mask for the detected sun.
        edge (numpy.ndarray or None): Detected edges in the image.
        sunLocation (numpy.ndarray or None): The location of the detected sun.
    Methods:
        __init__(self, imageFileName):
            Initializes the ImageProcessor with the given image path.
        load_image(self):
            Loads the image and converts it to grayscale.
        display_image(self):
            Displays the grayscale image with optional sun location and edge overlays.
        convert_to_array(self):
            Converts the grayscale image to a NumPy array.
        edgeDetection(self):
            Detects edges in the image using a combination of thresholding, contour detection, and Hough Circle Transform.
        detect_edges(self):
            Detects edges using the Canny edge detection method.
        sunDetectionSEP(self):
            Detects the sun in the image using Source Extractor and calculates its location and radius.
    """
    
    def __init__(self, input_data):
        
        """
        Initializes the ImageProcessor with either image data array or file name.

        Parameters:
        input_data (numpy.ndarray or str): The image data array or the file name.
        """
        if isinstance(input_data, str):
            self.imageFileName = input_data
            self.load_image()
        elif isinstance(input_data, np.ndarray):
            self.imageData = input_data
        else:
            raise ValueError("input_data must be a numpy array or a file name")
        self.logger = logging.getLogger('ImageProcessor')
        self.logger.setLevel(logging.INFO)
        self.logger.info("Initializing ImageProcessor")

        self.sun_mask = None
        self.edge = None
        self.sunLocation = None
        self.sunAlt = None
        self.sunAzi = None

        self.initial_latitude = None
        self.initial_longitude = None
        
    def load_image(self):
        """
        Loads an image from the specified path and converts it to grayscale.

        This method attempts to open an image file from the path stored in `self.imageFileName`.
        If successful, the image is converted to grayscale and stored in `self.image`.
        If the file is not found, an error message is printed.

        Raises:
            FileNotFoundError: If the image file does not exist at `self.imageFileName`.
        """
        try:
            if self.imageFileName.lower().endswith('.fits'):
                # Load FITS file
                imageData = fits.getdata(self.imageFileName)
                self.imageData = np.ascontiguousarray(np.flipud(np.fliplr(imageData.astype(float))))
            else:
                self.imageData = np.fliplr(Image.open(self.imageFileName).convert('L'))  # Convert to grayscale
                self.imageData = np.ascontiguousarray(self.imageData)
            #import pdb; pdb.set_trace()
            self.logger.info(f"Image loaded successfully: {self.imageFileName}")
        except FileNotFoundError:
            self.logger.error(f"Image not found at {self.imageFileName}.")

    def getInitialLocationFromGPS(self):

        
        gps = GPSReader(system='GNSS').connect()
        
        try:
            coords = gps.get_coordinates()
            if coords:
                lat, lon, gps_time, sats = coords
                local_time = GPSReader.convert_to_taipei_time(gps_time)
                self.initial_latitude = lat
                self.initial_longitude = lon
                self.logger.info(f"Latitude: {lat:.6f}, Longitude: {lon:.6f}")
            else:
                print("Could not get GPS fix")
        finally:
            gps.disconnect()

        return lat, lon

    def display_image(self):
        """Displays the grayscale image."""
        if self.imageData is not None:
            m, s = np.mean(self.imageData), np.std(self.imageData)

            plt.imshow(self.imageData,origin='lower', cmap='gray', vmin=m-s, vmax=m+s)  # Display in grayscale
            if self.sunLocation is not None:
                plt.scatter(self.sunLocation[0], self.sunLocation[1], s=100, c='red', marker='x')
            if self.edge is not None:
                for x, y, r in self.edge:
                    circle = plt.Circle((x, y), r, color='red', fill=False)
                    plt.gca().add_patch(circle)
            #if self.sunAlt is not None:
            #    c = plt.Circle((x, y), (1-self.sunAlt/90)*r, color='blue', fill=False)
            #    plt.gca().add_patch(c)

            #if self.sunAzi is not None:
            #    for x, y, r in self.edge[0,:]:
                #y = self.edge[0,1]
                #x, y, _ = self.edge[0,:]
            #        self.logger.info(f"Plotting north arrow based on Sun Azimuth: {self.sunAzi}")
            #        self.plot_north_arrow(plt.gca(), self.sunAzi, (x,y))

            plt.axis('off')
            plt.show()
        else:
            self.logger.warning("No image to display. Please load an image first.")


    def edgeDetection(self, display=False):

        mask = cv2.inRange(self.imageData, 50, self.imageData.max()/2)
        masked_image = cv2.bitwise_and(self.imageData, self.imageData, mask=mask)
        image_8bit = cv2.convertScaleAbs(masked_image, alpha=(255.0 / masked_image.max()))

        edges = cv2.Canny(image_8bit, image_8bit.mean(), image_8bit.mean()+image_8bit.std())
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=3)

        circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, dp=1, minDist=self.imageData.shape[0] // 2,
                                        param1=10, param2=20, 
                                        minRadius=1000, 
                                        maxRadius=1200)
        
        if circles is None:
            circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, dp=1, minDist=self.imageData.shape[0] // 2,
                                        param1=10, param2=20, 
                                        minRadius=800, 
                                        maxRadius=1200)
        
        # Final visualization of detected circles
        if display and circles is not None:
            plt.figure()
            plt.title('Detected Horizon Circles')
            m, s = np.mean(self.imageData), np.std(self.imageData)
            plt.imshow(self.imageData,origin='lower', cmap='gray', vmin=m - s, vmax=m)
            for x, y, r in circles[0,:]:
                circle = plt.Circle((x, y), r, color='red', fill=False)
                plt.gca().add_patch(circle)
            plt.axis('off')
            plt.show()
        
        self.edge = circles[0,:]
        return circles[0,:]
    

    def sunDetectionSEP(self, imageData = None, display=False):
        threshold = 400
        step = 50
        min_threshold = 50

        if imageData is None:
            imageData = self.imageData
        m, s = np.mean(imageData), np.std(imageData)
    
        bkg = sep.Background(imageData.astype(float))
    
        while threshold >= min_threshold:
            try:
                # Attempt to extract sources with the current threshold
                objects = sep.extract(imageData, threshold, err=bkg.globalrms)
                
                # If we find objects, break the loop
                if len(objects) > 0:
                    break
            except Exception as e:
                # Catch any exception and print the error message
                print(f"Error while detecting objects at threshold {threshold}: {str(e)}")
                print("Skipping this frame.")
                return None
            
            # If no objects are found, decrease the threshold
            threshold -= step
    
        # Ensure at least one object was found
        if len(objects) == 0:
            #print("No objects found at any threshold")
            return None
    
        # Get the object with the maximum flux
        cat = max(objects, key=lambda obj: obj['flux'])
        r = np.sqrt(cat['a']**2 + cat['b']**2)
        circle = np.array([cat['x'], cat['y'], r])
        self.sunLocation = circle
        if display is True:
            self.display_image()
        
        return circle
    

    def getLocalTimeFromFileName(self, file_path):
        """
        Get the local time from the system clock.
        
        Returns:
            datetime: The current local time.
        """
        base_filename = os.path.basename(file_path)

        # Extract the date and time part from the filename
        # Assuming the format is 'image_<YYYYMMDD>_<HH_MM_SS_MS>.fits'
        # Splitting by underscores and getting the relevant parts
        parts = base_filename.split('_')
        date_str = parts[1]  # '20241129'
        time_str = parts[2] + ':' + parts[3] + ':' + parts[4]  # '13:50:45'

        # Combine date and time strings
        combined_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str}"  # '2024-11-29 13:50:45'

        # Convert to datetime object
        dt = datetime.strptime(combined_str, '%Y-%m-%d %H:%M:%S')
        
        return dt
   
    def calculateSun(self, local_latitude, local_longitude, local_time):
        # Load ephemeris data
        planets = load('de421.bsp')
        earth = planets['earth']
        sun = planets['sun']
        ts = load.timescale()

        # Input data for sanity check
        #latitude = 25.013773     # Latitude in degrees (example: Taipei City)
        #longitude = 121.852062  # Longitude in degrees (example: Taipei City)
        latitude = local_latitude     # Latitude in degrees
        longitude = local_longitude  # Longitude in degrees
        timezone_offset = +8  # Offset from UTC in hours (example: PDT for Taipei City is UTC+8)

        # Convert local time to UTC
        utc_time = local_time - timedelta(hours=timezone_offset)
        time = ts.utc(utc_time.year, utc_time.month, utc_time.day,
                    utc_time.hour, utc_time.minute, utc_time.second)

        # Observer's location
        location = Topos(latitude_degrees=latitude, longitude_degrees=longitude)

        # Calculate the Sun's position from this location and time
        observer = earth + location
        astrometric = observer.at(time).observe(sun)
        altitude, azimuth, _ = astrometric.apparent().altaz()

        self.sunAlt = altitude.degrees
        self.sunAzi = azimuth.degrees
        #return altitude.degrees, azimuth.degrees
        # Output the altitude and azimuth
        self.logger.info(f"Sun's Actual Altitude: {altitude.degrees:.2f} Azimuth: {azimuth.degrees:.2f}°")

    def calSunAltAzi(self, zenith, sun_position, radius_pixels):
        """
        Calculate the altitude and azimuth of the Sun based on pixel positions in an all-sky image.

        Parameters:
            zenith (tuple): (x, y) pixel coordinates of the zenith.
            sun_position (tuple): (x, y) pixel coordinates of the Sun.
            radius_pixels (float): The radius of the all-sky image in pixels (horizon radius).

        Returns:
            altitude (float): Altitude of the Sun in degrees.
            azimuth (float): Azimuth of the Sun in degrees (0° = North, 90° = East).
        """
        # Extract coordinates
        x_zenith, y_zenith = zenith
        x_sun, y_sun = sun_position

        # Calculate the pixel distance between zenith and Sun
        dx = x_sun - x_zenith
        dy = y_sun - y_zenith
        pixel_distance = np.sqrt(dx**2 + dy**2)

        # Calculate altitude (angle above the horizon)
        # Altitude is proportional to (1 - normalized distance from zenith)
        normalized_distance = pixel_distance / radius_pixels
        altitude = 90 - normalized_distance * 90  # Map [0, 1] to [90°, 0°]

        # Calculate azimuth (angle clockwise from North)
        azimuth = (np.degrees(np.arctan2(dx, dy)) + 360) % 360  # Adjust to [0, 360)


        self.sunMeasuredAlt = altitude
        self.sunMeasuredAzi = azimuth

        return altitude, azimuth

    def calculateLatLon(self, altitude, azimuth, local_time):
        planets = load('de421.bsp')
        earth = planets['earth']
        sun = planets['sun']
        ts = load.timescale()

        def compute_sun_alt_az(latitude, longitude, local_time, timezone_offset=8):
            # Generate observer's location
            location = Topos(latitude_degrees=latitude, longitude_degrees=longitude)
            utc_time = local_time - timedelta(hours=timezone_offset)

            time = ts.utc(utc_time.year, utc_time.month, utc_time.day,
                            utc_time.hour, utc_time.minute, utc_time.second)
            
            # Compute Sun's apparent position from this location and time
            observer = earth + location
            astrometric = observer.at(time).observe(sun)
            alt, az, _ = astrometric.apparent().altaz()
            
            return alt.degrees, az.degrees
        
        def objective(coords):
            latitude, longitude = coords
            computed_alt, computed_az = compute_sun_alt_az(latitude, longitude, local_time)
            alt_diff = (computed_alt - altitude)**2
            az_diff = (computed_az - azimuth)**2
            return alt_diff + az_diff

        # Initial guess (example for somewhere near the equator)
        initial_guess = [0.0, 0.0]  # latitude, longitude in degrees

        # Minimize the objective function
        result = minimize(objective, initial_guess, bounds=[(-90, 90), (-180, 180)])

        # Extract latitude and longitude if successful
        if result.success:
            latitude, longitude = result.x
            self.logger.info(f"Latitude: {latitude:.6f}°, Longitude: {longitude:.6f}°")
        else:
            self.logger.info("Optimization failed.")
            latitude, longitude = None, None
        return latitude, longitude


def testFITSimage(imageFileName):
    processor = ImageProcessor(imageFileName) 
    localTime = processor.getLocalTimeFromFileName(imageFileName)
    processor.initial_latitude = 24.852314
    processor.initial_longitude = 120.923478
    self.logger.info(f"Local Time: {localTime}")

    processor.calculateSun(processor.initial_latitude, processor.initial_longitude, localTime)

    #processor.load_image()
    processor.sunDetectionSEP(display=True)
    edges = processor.edgeDetection(display=True)
    sun_x, sun_y, sun_r = processor.sunLocation
    allsky_x, allsky_y, allsky_r = edges[0,0], edges[0,1], edges[0,2]
    
    self.logger.info(f"{processor.calSunAltAzi((allsky_x, allsky_y), (sun_x, sun_y), allsky_r)}")
    deltaAlt = 0
    deltaAzi = 0

    Alt = processor.sunMeasuredAlt + deltaAlt
    Azi = processor.sunMeasuredAzi + deltaAzi
    
    self.logger.info(f"Altitude: {Alt} Delta Azimuth: {Azi}")
    #latitude, longitude = processor.calculateLatLon(Alt, Azi, processor.getLocalTimeFromFileName(imageFileName))
    #self.logger.info(f"Calculated Latitude: {latitude} Longitude: {longitude}")


def initCalibrate(imageFileName):
    #imageFileName = os.path.join(image_dir, 'image_5_2024-09-06_16-19-27.jpg')   
    processor = ImageProcessor(imageFileName) 
    localTime = processor.getLocalTimeFromFileName(imageFileName)
    self.logger.info(f"Local Time: {localTime}")
    processor.initial_latitude = 24.874241
    processor.initial_longitude = 120.947295

    processor.calculateSun(processor.initial_latitude, processor.initial_longitude, localTime)
    

    processor.sunDetectionSEP(display=False)
    edges = processor.edgeDetection(display=False)
    sun_x, sun_y, sun_r = processor.sunLocation
    allsky_x, allsky_y, allsky_r = edges[0,0], edges[0,1], edges[0,2]
    
    self.logger.info(f"{processor.calSunAltAzi((allsky_x, allsky_y), (sun_x, sun_y), allsky_r)}")
    
    deltaAlt = processor.sunAlt - processor.sunMeasuredAlt
    deltaAzi = processor.sunAzi - processor.sunMeasuredAzi
    self.logger.info(f"Delta Altitude: {deltaAlt} Delta Azimuth: {deltaAzi}")
    
    return deltaAlt, deltaAzi, edges

def measureSun(imageFileName, deltaAlt, deltaAzi, edges):
    
    processor = ImageProcessor(imageFileName)
    processor.sunDetectionSEP()
    edges = processor.edgeDetection()

    sun_x, sun_y, sun_r = processor.sunLocation
    processor.edge = edges
    allsky_x, allsky_y, allsky_r = edges[0,0], edges[0,1], edges[0,2]
    self.logger.info(f"Sun Location: {processor.sunLocation}")
    self.logger.info(f"Horizon: {edges}")
    self.logger.info(f"{processor.calSunAltAzi((allsky_x, allsky_y), (sun_x, sun_y), allsky_r)}")
    
    Alt = processor.sunMeasuredAlt + deltaAlt
    Azi = processor.sunMeasuredAzi + deltaAzi
    
    self.logger.info(f"Altitude: {Alt} Azimuth: {Azi}")
    latitude, longitude = processor.calculateLatLon(Alt, Azi, processor.getLocalTimeFromFileName(imageFileName))
    self.logger.info(f"Calculated Latitude: {latitude} Longitude: {longitude}")

    processor.display_image()
    return Alt, Azi, latitude, longitude

def runTest():
    # Path to the directory containing images
    image_dir = './images'
    #file_pattern = "./images/20241129/image_20241129_14_40_1[456]_*.fits"
    #file_pattern = "./images/20241129/image_20241129_14_38_59_*.fits"
    #matching_files = glob.glob(file_pattern)
    file_pattern = "./images/20241129/image_20241129_14_39_00_*.fits"
    
    #file_pattern = './output/image_20250204_14_5*.fits'
    matching_files = glob.glob(file_pattern)
    print(matching_files)
    matching_files.sort()
    deltaAlt, deltaAzi, edges = initCalibrate(matching_files[0])
    
    # Initialize lists to store the results
    altitudes = []
    azimuths = []
    latitudes = []
    longitudes = []
    
    for filename in matching_files[1:]:
        altitude, azimuth, lat, lon = measureSun(filename, deltaAlt, deltaAzi, edges)
        altitudes.append(altitude)
        azimuths.append(azimuth)
        latitudes.append(lat)
        longitudes.append(lon)

    # Convert lists to numpy arrays
    altitudes = np.array(altitudes)
    azimuths = np.array(azimuths)
    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)

    # Calculate means
    mean_altitude = np.mean(altitudes)
    mean_azimuth = np.mean(azimuths)
    mean_latitude = np.mean(latitudes)
    mean_longitude = np.mean(longitudes)

    # Calculate standard deviations
    std_altitude = np.std(altitudes)
    std_azimuth = np.std(azimuths)
    std_latitude = np.std(latitudes)
    std_longitude = np.std(longitudes)

    print(f"Mean Altitude: {mean_altitude:.2f}° ± {std_altitude:.3f}°")
    print(f"Mean Azimuth: {mean_azimuth:.2f}° ± {std_azimuth:.3f}°")
    print(f"Mean Latitude: {mean_latitude:.6f}° ± {std_latitude:.6f}°")
    print(f"Mean Longitude: {mean_longitude:.6f}° ± {std_longitude:.6f}°")


if __name__ == "__main__":

    runTest()