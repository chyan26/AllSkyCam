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


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - [Line: %(lineno)d]')

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
    
    def __init__(self, imageFileName):
        self.imageFileName = imageFileName
        self.image = None
        self.image_array = None
        self.sun_mask = None


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
            self.image = np.fliplr(Image.open(self.imageFileName).convert('L'))  # Convert to grayscale
            logging.info(f"Image loaded successfully: {self.imageFileName}")
        except FileNotFoundError:
            logging.error(f"Image not found at {self.imageFileName}.")

    def plot_north_arrow(self, ax, sun_azimuth, image_center, arrow_length=500.0, color='blue'):
        """
        Plot an arrow pointing north based on the Sun's azimuth on an all-sky image.

        Parameters:
            ax (matplotlib.axes.Axes): The matplotlib axes to plot on.
            sun_azimuth (float): The Sun's azimuth angle in degrees (0° = North, 90° = East).
            image_center (tuple): The (x, y) coordinates of the image center.
            arrow_length (float): Length of the arrow (relative to plot size).
            color (str): Color of the arrow.
        """
        # Convert azimuth to radians for plotting
        azimuth_rad = np.radians((90 - sun_azimuth) % 360)

        # Calculate arrow endpoint based on azimuth angle
        x_end = image_center[0] + arrow_length * np.cos(azimuth_rad)
        y_end = image_center[1] + arrow_length * np.sin(azimuth_rad)

        # Plot the arrow from the center point in the direction of the North
        ax.arrow(
            image_center[0], image_center[1],  # Starting point of the arrow (image center)
            x_end - image_center[0], y_end - image_center[1],  # Arrow direction and length
            head_width=100.0, head_length=70.0, fc=color, ec=color
        )

    def display_image(self):
        """Displays the grayscale image."""
        if self.image is not None:
            plt.imshow(self.image_array, cmap='gray')  # Display in grayscale
            if self.sunLocation is not None:
                plt.scatter(self.sunLocation[0], self.sunLocation[1], s=100, c='red', marker='x')
            if self.edge is not None:
                for x, y, r in self.edge[0,:]:
                    circle = plt.Circle((x, y), r, color='red', fill=False)
                    plt.gca().add_patch(circle)
            if self.sunAlt is not None:
                c = plt.Circle((x, y), (1-self.sunAlt/90)*r, color='blue', fill=False)
                plt.gca().add_patch(c)

            if self.sunAzi is not None:
                for x, y, r in self.edge[0,:]:
                #y = self.edge[0,1]
                #x, y, _ = self.edge[0,:]
                    logging.info(f"Plotting north arrow based on Sun Azimuth: {self.sunAzi}")
                    self.plot_north_arrow(plt.gca(), self.sunAzi, (x,y))

            plt.axis('off')
            plt.show()
        else:
            logging.warning("No image to display. Please load an image first.")

    def convert_to_array(self):
        """Converts the grayscale image to a NumPy array."""
        if self.image is not None:
            self.image_array = np.array(self.image)
            logging.info("Image converted to NumPy array.")
            return self.image_array
        else:
            logging.warning("No image to convert. Please load an image first.")
            return None

    def edgeDetection(self):
        min_sun_value = 200
        max_sun_value = 255

        # Create a binary mask of non-black pixels
        _, binary_mask = cv2.threshold(self.image_array, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (should be the fisheye circle)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit an ellipse to the largest contour
        (x, y), (MA, ma), angle = cv2.fitEllipse(largest_contour)

        logging.info(f"Ellipse center: ({x}, {y}), Major Axis: {MA}, Minor Axis: {ma}, Angle: {angle}")
        # Calculate the average radius
        radius = int((MA + ma) / 4)  # Divide by 4 because MA and ma are diameters
        
        # Create a circular mask
        mask = np.zeros_like(self.image_array)
        cv2.circle(mask, (int(x), int(y)), int(radius * 1.1), 255, -1)
        
        # Apply the mask to the original image
        masked_img = cv2.bitwise_and(self.image_array, mask)
        
        # Apply edge detection only to the masked region
        blurred = cv2.GaussianBlur(masked_img, (9, 9), 2)
        edges = cv2.Canny(blurred, 30, 60)
        
        # Dilate the edges
        kernel = np.ones((10,10), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find circles using Hough Circle Transform with tighter parameters
        circles = cv2.HoughCircles(
            dilated,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.image_array.shape[0]/2,
            param1=50,
            param2=30,
            minRadius=int(radius * 0.9),  # Use fitted ellipse to constrain radius search
            maxRadius=int(radius * 1.1)
        )
        
        self.edge = circles
        return circles
    
    def sunDetectionSEP(self):
        m, s = np.mean(self.image_array), np.std(self.image_array)

        bkg = sep.Background(self.image_array.astype(float))
        #data_sub = self.image_array - bkg.rms()

        objects = sep.extract(self.image_array, 250, err=bkg.globalrms)
        if len(objects) == 0:
            objects = sep.extract(self.image_array, 200, err=bkg.globalrms)
            
        ind=np.argmax(objects['flux'])
        
        cat = objects[ind:ind+1]
        r = np.sqrt(cat['a']**2+cat['b']**2)
        circle = np.array([cat['x'][0],cat['y'][0], r[0]])
        
        #plt.show()
        self.sunLocation = circle
        #return circle
    
    def image_to_math_azimuth(image_azimuth):
        """
        Convert azimuth from all-sky image system to mathematical system.
        
        Parameters:
            image_azimuth (float): Azimuth in the all-sky image system (0° = North, 90° = East, clockwise).
            
        Returns:
            float: Azimuth in the mathematical system (0° = East, 90° = North, counterclockwise).
        """
        math_azimuth = (90 - image_azimuth) % 360
        return math_azimuth

    def math_to_image_azimuth(math_azimuth):
        """
        Convert azimuth from mathematical system to all-sky image system.
        
        Parameters:
            math_azimuth (float): Azimuth in the mathematical system (0° = East, 90° = North, counterclockwise).
            
        Returns:
            float: Azimuth in the all-sky image system (0° = North, 90° = East, clockwise).
        """
        image_azimuth = (90 - math_azimuth) % 360
        return image_azimuth

    def getLocalTimeFromFileName(self, file_path):
        """
        Get the local time from the system clock.
        
        Returns:
            datetime: The current local time.
        """
        base_filename = os.path.basename(file_path)

        # Extract the date and time part from the filename
        # Assuming the format is 'image_<index>_<YYYY-MM-DD>_<HH-MM-SS>.png'
        # Splitting by underscores and getting the relevant parts
        parts = base_filename.split('_')
        date_str = parts[2]  # '2024-09-06'
        time_str = parts[3].split('.')[0]  # '16-18-09' from '16-18-09.png'

        # Combine date and time strings
        combined_str = f"{date_str} {time_str.replace('-', ':')}"  # Replace '-' in time with ':'

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
        logging.info(f"Sun's Actual Altitude: {altitude.degrees:.2f} Azimuth: {azimuth.degrees:.2f}°")

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
        azimuth = (np.degrees(np.arctan2(dx, -dy)) + 360) % 360  # Adjust to [0, 360)


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
            print(f"Latitude: {latitude:.6f}°, Longitude: {longitude:.6f}°")
        else:
            print("Optimization failed.")

        return latitude, longitude


def initCalibrate(imageFileName):
    image_dir = './images'
    imageFileName = os.path.join(image_dir, 'image_5_2024-09-06_16-19-27.jpg')   
    processor = ImageProcessor(imageFileName) 
    localTime = processor.getLocalTimeFromFileName(imageFileName)
    logging.info(f"Local Time: {localTime}")
    processor.initial_latitude = 25.013773
    processor.initial_longitude = 121.852062

    processor.calculateSun(processor.initial_latitude, processor.initial_longitude, localTime)
    

    processor.load_image()
    image_data = processor.convert_to_array()
    processor.sunDetectionSEP()
    edges = processor.edgeDetection()
    sun_x, sun_y, sun_r = processor.sunLocation
    allsky_x, allsky_y, allsky_r = edges[0][0][0], edges[0][0][1], edges[0][0][2]
    
    logging.info(f"{processor.calSunAltAzi((allsky_x, allsky_y), (sun_x, sun_y), allsky_r)}")
    
    deltaAlt = processor.sunAlt - processor.sunMeasuredAlt
    deltaAzi = processor.sunAzi - processor.sunMeasuredAzi
    logging.info(f"Delta Altitude: {deltaAlt} Delta Azimuth: {deltaAzi}")
    
    return deltaAlt, deltaAzi

def measureSun(imageFileName, deltaAlt, deltaAzi):
    image_dir = './images'
    imageFileName = os.path.join(image_dir, imageFileName)   
    
    processor = ImageProcessor(imageFileName)

    processor.load_image()
    image_data = processor.convert_to_array()
    processor.sunDetectionSEP()
    edges = processor.edgeDetection()
    sun_x, sun_y, sun_r = processor.sunLocation
    allsky_x, allsky_y, allsky_r = edges[0][0][0], edges[0][0][1], edges[0][0][2]
    
    logging.info(f"{processor.calSunAltAzi((allsky_x, allsky_y), (sun_x, sun_y), allsky_r)}")
    
    Alt = processor.sunMeasuredAlt + deltaAlt
    Azi = processor.sunMeasuredAzi + deltaAzi
    
    logging.info(f"Altitude: {Alt} Delta Azimuth: {Azi}")
    latitude, longitude = processor.calculateLatLon(Alt, Azi, processor.getLocalTimeFromFileName(imageFileName))
    logging.info(f"Calculated Latitude: {latitude} Longitude: {longitude}")

    #return Alt, Azi

def test():
    # Path to the directory containing images
    image_dir = './images'
    
    # Loop through each file in the directory
    for filename in os.listdir(image_dir)[0:2]:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Add more formats if needed
            imageFileName = os.path.join(image_dir, filename)
            logging.info(f"Processing image: {imageFileName}")

            # Initialize the ImageProcessor
            processor = ImageProcessor(imageFileName)

            # Load the image and convert it to a NumPy array
            processor.load_image()
            image_data = processor.convert_to_array()
            
            localTime = processor.getLocalTimeFromFileName(imageFileName)
            logging.info(f"Local Time: {localTime}")
            processor.initial_latitude = 25.013773
            processor.initial_longitude = 121.852062

            processor.calculateSun(processor.initLat, processor.initial_longitude, localTime)
            
            if image_data is not None:
                logging.info(f"Image shape: {image_data.shape}")

            processor.sunDetectionSEP()

            # Perform edge detection
            edges = processor.edgeDetection()
            logging.info(f"The detected circle of horizon (x, y, r )= {edges}")

            logging.info(f"Sun Location: {processor.sunLocation}")
            sun_x, sun_y, sun_r = processor.sunLocation
            allsky_x, allsky_y, allsky_r = edges[0][0][0], edges[0][0][1], edges[0][0][2]

            distance = np.sqrt((sun_x - allsky_x)**2 + (sun_y - allsky_y)**2)
            logging.info(f"Distance between sun and allsky center: {distance}")
            logging.info(f"Convert to Azimuth = {90-(distance/allsky_r)*90}")


            logging.info(f"{processor.calSunAltAzi((allsky_x, allsky_y), (sun_x, sun_y), allsky_r)}")
            # Display the original image    
            processor.display_image()

if __name__ == "__main__":
    deltaAlt, deltaAzi = initCalibrate('image_5_2024-09-06_16-19-27.jpg')
    measureSun('image_6_2024-09-06_16-19-28.jpg', deltaAlt, deltaAzi)
    #logging.info(f"Current Sun Altitude: {currentSunAlt} Current Sun Azimuth: {currentSunAzi}")