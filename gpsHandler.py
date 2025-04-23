import threading
import time
import logging
from ubloxReader import GPSReader

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
        
        # Thread control
        self.running = False
        self.thread = None
        self.lock = threading.RLock()
        
        # Events
        self.first_fix_event = threading.Event()
    
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
                            self.satellites = sats
                            self.gps_fix = True
                            
                        # Signal first fix if not already done
                        if not self.first_fix_event.is_set():
                            logger.info(f"First GPS fix obtained: Lat={lat:.6f}, Lon={lon:.6f}, Satellites={sats}")
                            self.first_fix_event.set()
                            
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