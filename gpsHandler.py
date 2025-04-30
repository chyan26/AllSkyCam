import threading
import time
import logging
import math
from collections import deque
from ubloxReader import GPSReader
import datetime
import subprocess

logger = logging.getLogger("gps_handler")
system_logger = logging.getLogger("idsExposure.py")  # Use the main system logger

class GPSHandler:
    """
    Dedicated class to handle GPS updates at a specified frequency.
    Maintains current GPS position and provides thread-safe access.
    """
    def __init__(self, update_frequency_hz=10, default_lat=None, default_lon=None, log_to_system=True):
        """
        Initialize GPSHandler.
        
        Args:
            update_frequency_hz: Frequency of GPS updates in Hz (default: 10Hz)
            default_lat: Default latitude to use if GPS not available
            default_lon: Default longitude to use if GPS not available
            log_to_system: Whether to log each GPS update to the system log
        """
        # Convert frequency in Hz to time interval in seconds
        self.update_interval = 1.0 / update_frequency_hz  # Convert Hz to seconds between updates
        self.default_lat = default_lat
        self.default_lon = default_lon
        self.log_to_system = log_to_system
        
        # GPS state variables
        self.latitude = default_lat
        self.longitude = default_lon
        self.timestamp = None
        self.satellites = 0
        self.connected = False
        self.gps_fix = False
        
        # Heading calculation variables
        self.heading = None
        self.heading_callback = None  # Callback for heading updates
        self.position_history = deque(maxlen=10)  # Store 10 positions (1 second of data at 10Hz)
        self.last_heading_calc_time = 0
        self.heading_calc_interval = 1.0  # Calculate heading every 1 second
        
        # Thread control
        self.running = False
        self.thread = None
        self.lock = threading.RLock()
        
        # Events
        self.first_fix_event = threading.Event()

        self.speed = 0  # Speed in m/s
        self.speed_callback = None  # Callback for speed updates

    def start(self):
        """Start the GPS update thread."""
        if self.running:
            logger.warning("GPS thread already running")
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        logger.info(f"GPS update thread started (target: {1/self.update_interval}Hz)")
        return True
    
    def stop(self):
        """Stop the GPS update thread."""
        if not self.running:
            logger.info("GPS thread not running")
            return
            
        logger.info("Stopping GPS thread...")
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                logger.warning("GPS thread did not stop within timeout")
        
        self.thread = None
        logger.info("GPS update thread stopped")
    
    def _update_loop(self):
        """Main GPS update loop that runs in a separate thread."""
        gps = None
        try:
            # Connect to GPS with optimal settings for higher update rate
            gps = GPSReader(system='GNSS', baudrate=115200).connect()
            
            with self.lock:
                self.connected = True
            
            logger.info("GPS Reader connected in thread")
            
            while self.running:
                loop_start = time.time()
                
                coords = gps.get_coordinates()
                if coords:
                    lat, lon, gps_time, sats = coords
                    
                    # Basic validation
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        with self.lock:
                            self.latitude = lat
                            self.longitude = lon
                            self.timestamp = gps_time
                            
                            # Ensure gps_time is a valid datetime object
                            if isinstance(gps_time, datetime.datetime):
                                self.timestamp = gps_time
                            elif isinstance(gps_time, (int, float)):  # If it's a Unix timestamp
                                self.timestamp = datetime.datetime.fromtimestamp(gps_time)
                            elif isinstance(gps_time, datetime.time):  # If it's a time object
                                today = datetime.date.today()
                                self.timestamp = datetime.datetime.combine(today, gps_time)
                            elif isinstance(gps_time, str):  # If it's a string time format
                                try:
                                    # Try to parse common time formats
                                    time_obj = datetime.datetime.strptime(gps_time, "%H:%M:%S").time()
                                    today = datetime.date.today()
                                    self.timestamp = datetime.datetime.combine(today, time_obj)
                                except ValueError:
                                    logger.warning(f"Invalid GPS time format: {gps_time}")
                                    self.timestamp = None
                            else:
                                logger.warning(f"Invalid GPS time format: {gps_time}")
                                self.timestamp = None
                            
                            self.satellites = sats
                            self.gps_fix = True
                            
                            # Store position for heading calculation
                            self.position_history.append((lat, lon, gps_time))

                        system_logger.info(f"GPS timestamp updated: {gps_time}")

                        # Signal first fix if not already done
                        if not self.first_fix_event.is_set():
                            logger.info(f"First GPS fix obtained: Lat={lat:.6f}, Lon={lon:.6f}, Satellites={sats}")
                            self.first_fix_event.set()

                        # Calculate heading every 1 second
                        current_time = time.time()
                        if current_time - self.last_heading_calc_time >= self.heading_calc_interval:
                            self._calculate_heading()
                            self.last_heading_calc_time = current_time
                            
                        # Log to system log at the same frequency as GPS updates
                        if self.log_to_system:
                            system_logger.info(f"Updated GPS location: Latitude={lat:.6f}, Longitude={lon:.6f}, Time={gps_time}, Satellites={sats}")
                    else:
                        logger.warning(f"Received invalid GPS coordinates: {lat}, {lon}")
                
                # Calculate sleep time to maintain target frequency
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.update_interval - elapsed)
                
                if sleep_time < 0.001:  # Less than 1ms remaining
                    logger.debug("GPS processing took longer than update interval")
                    
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Error in GPS update thread: {e}", exc_info=True)
            with self.lock:
                self.connected = False
                self.gps_fix = False
        finally:
            if gps:
                gps.disconnect()
                logger.info("GPS Reader disconnected")
            
            with self.lock:
                self.connected = False
                
    def set_speed_callback(self, callback):
        """Set a callback function to be called when speed is updated."""
        with self.lock:
            self.speed_callback = callback

    def _calculate_heading(self):
        """
        Calculate heading and speed based on position history.
        Only calculates if we have sufficient data points.
        """
        with self.lock:
            if len(self.position_history) < 2:
                return
            
            # Get most recent positions
            positions = list(self.position_history)
            latest = positions[-1]
            previous = positions[-2]
            
            # Extract coordinates and timestamps
            lat1, lon1, time1 = previous
            lat2, lon2, time2 = latest
            
            # Skip if positions are identical
            if abs(lat1 - lat2) < 1e-9 and abs(lon1 - lon2) < 1e-9:
                return
            
            # Calculate time difference in seconds
            try:
                time_diff = self._convert_to_timestamp(time2) - self._convert_to_timestamp(time1)
                if time_diff <= 0:
                    return
            except Exception as e:
                logger.error(f"Error processing timestamps for speed calculation: {e}")
                return
            
            # Calculate distance using the Haversine formula
            distance = self._haversine(lat1, lon1, lat2, lon2)
            
            # Calculate speed in m/s and convert to km/h
            speed_mps = distance / time_diff
            self.speed = speed_mps * 3.6  # Convert to km/h
            
            if self.speed_callback:
                self.speed_callback(self.speed)

            # Calculate heading
            lat1_rad = math.radians(lat1)
            lon1_rad = math.radians(lon1)
            lat2_rad = math.radians(lat2)
            lon2_rad = math.radians(lon2)
            
            y = math.sin(lon2_rad - lon1_rad) * math.cos(lat2_rad)
            x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
                math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(lon2_rad - lon1_rad)
            heading_rad = math.atan2(y, x)
            
            heading_deg = (math.degrees(heading_rad) + 360) % 360
            
            # Update heading
            old_heading = self.heading
            self.heading = heading_deg
            
            # Log significant changes in heading
            if old_heading is None or abs(old_heading - heading_deg) > 5:
                logger.info(f"GPS Heading updated: {heading_deg:.1f}° ({self._heading_to_cardinal(heading_deg)})")
                system_logger.info(f"GPS Heading: {heading_deg:.1f}° ({self._heading_to_cardinal(heading_deg)})")
            
            # Call the heading callback if set
            if self.heading_callback is not None:
                try:
                    self.heading_callback(heading_deg)
                except Exception as e:
                    logger.error(f"Error in heading callback: {e}")

    def _convert_to_timestamp(self, time_obj):
        """
        Convert a datetime-like object to a Unix timestamp.
        """
        if isinstance(time_obj, (int, float)):
            return time_obj  # Already a timestamp
        elif hasattr(time_obj, 'timestamp'):
            return time_obj.timestamp()  # datetime.datetime
        elif isinstance(time_obj, datetime.time):
            # Handle datetime.time (assume same day for simplicity)
            today = datetime.date.today()
            return time.mktime(datetime.datetime.combine(today, time_obj).timetuple())
        else:
            raise TypeError(f"Unsupported timestamp type: {type(time_obj)}")
    
    def _haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great-circle distance between two points on the Earth using the Haversine formula.
        
        Args:
            lat1, lon1: Latitude and longitude of the first point in degrees
            lat2, lon2: Latitude and longitude of the second point in degrees
        
        Returns:
            Distance in meters
        """
        R = 6371000  # Radius of Earth in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def _heading_to_cardinal(self, heading):
        """Convert heading in degrees to cardinal direction."""
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
                     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        index = round(heading / 22.5) % 16
        return directions[index]
    
    def get_heading(self):
        """Thread-safe method to get current GPS heading."""
        with self.lock:
            return self.heading
    
    def set_heading_callback(self, callback):
        """
        Set a callback function that will be called whenever the GPS heading is updated.
        
        Args:
            callback: Function that takes a heading value in degrees as its argument
        """
        with self.lock:
            self.heading_callback = callback
    
    def get_coordinates(self):
        """Thread-safe method to get current GPS coordinates."""
        with self.lock:
            return (self.latitude, self.longitude, self.timestamp, self.satellites)
    
    def has_fix(self):
        """Check if GPS has a valid fix."""
        with self.lock:
            return self.gps_fix
    
    def wait_for_fix(self, timeout=10):
        """
        Wait for GPS to obtain a fix.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            Tuple of (latitude, longitude) or None if no fix within timeout
        """
        if self.first_fix_event.wait(timeout):
            return self.get_coordinates()[:2]  # Just return lat, lon
        
        # If timeout occurred, return default values if available
        if self.default_lat is not None and self.default_lon is not None:
            logger.info(f"Using default coordinates: Lat={self.default_lat}, Lon={self.default_lon}")
            return (self.default_lat, self.default_lon)
        
        return None

    def _haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great-circle distance between two points on the Earth's surface.
        """
        R = 6371e3  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c

    def get_speed(self):
        """Thread-safe method to get the current speed in km/h."""
        with self.lock:
            return self.speed
        
    def get_gps_time(self):
        """
        Thread-safe method to get the current GPS time.
        
        Returns:
            datetime.datetime: The current GPS time, or None if not available.
        """
        with self.lock:
            if self.timestamp:
                # Convert the timestamp to a datetime object if it's not already
                if isinstance(self.timestamp, datetime.datetime):
                    return self.timestamp
                else:
                    return datetime.datetime.fromtimestamp(self.timestamp)
            return None
        
