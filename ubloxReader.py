import serial
import logging
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Literal
from datetime import time, datetime
import pytz
import math

GNSSSystems = Literal['GPS', 'GLONASS', 'GALILEO', 'BEIDOU', 'GNSS']

@dataclass
class GPSData:
    latitude: float
    longitude: float
    altitude: float
    satellites: int
    quality: int
    time: Optional[time] = None
    speed_knots: float = 0.0
    course: float = 0.0

class GPSReader:
    def __init__(self, device='/dev/ttyACM0', baudrate=9600, 
                 timeout=1, system: GNSSSystems = 'GNSS'):
        self.device = device
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self._data = {}
        self._sentence_prefix = {
            'GPS': '$GP',
            'GLONASS': '$GL',
            'GALILEO': '$GA',
            'BEIDOU': '$BD',
            'GNSS': '$GN'
        }[system]

    def connect(self):
        self.serial_conn = serial.Serial(self.device, self.baudrate, timeout=self.timeout)
        return self

    def disconnect(self):
        if self.serial_conn:
            self.serial_conn.close()
            self.serial_conn = None

    def _convert_coordinates(self, coord_str: str, direction: str) -> float:
        if not coord_str:
            return 0.0
        
        try:
            degrees = float(coord_str[:2] if direction in 'NS' else coord_str[:3])
            minutes = float(coord_str[2:] if direction in 'NS' else coord_str[3:])
            decimal = degrees + minutes/60.0
            return -decimal if direction in ['S', 'W'] else decimal
        except ValueError:
            return 0.0

    def _parse_gga(self, parts: list):
        if len(parts) >= 15:
            try:
                time_str = parts[1]
                if time_str:
                    hours = int(time_str[0:2])
                    minutes = int(time_str[2:4])
                    seconds = int(float(time_str[4:]))
                    current_time = time(hours, minutes, seconds)
                else:
                    current_time = None

                lat = self._convert_coordinates(parts[2], parts[3])
                lon = self._convert_coordinates(parts[4], parts[5])
                
                self._data.update({
                    'time': current_time,
                    'latitude': lat,
                    'longitude': lon,
                    'altitude': float(parts[9]) if parts[9] else 0.0,
                    'satellites': int(parts[7]) if parts[7] else 0,
                    'quality': int(parts[6]) if parts[6] else 0
                })
            except (ValueError, IndexError):
                pass

    def _parse_vtg(self, parts: list):
        if len(parts) >= 9:
            self._data['speed_knots'] = float(parts[5]) if parts[5] else 0.0
            self._data['course'] = float(parts[1]) if parts[1] else 0.0

    def read_location(self) -> Optional[GPSData]:
        if not self.serial_conn:
            return None

        try:
            line = self.serial_conn.readline().decode('ascii', errors='replace').strip()
            if not line:
                return None

            parts = line.split(',')
            sentence = parts[0]

            if sentence.startswith(f'{self._sentence_prefix}GGA'):
                self._parse_gga(parts)
            elif sentence.startswith(f'{self._sentence_prefix}VTG'):
                self._parse_vtg(parts)

            if all(k in self._data for k in ['latitude', 'longitude', 'altitude', 'satellites', 'quality']):
                return GPSData(**self._data)
            
        except (ValueError, IndexError) as e:
            logging.error(f"Parse error: {e}")
        
        return None

    def get_coordinates(self, timeout_seconds: int = 10) -> Union[Tuple[float, float, time, int], None]:
        start_time = datetime.now()
        
        try:
            while (datetime.now() - start_time).seconds < timeout_seconds:
                location = self.read_location()
                if location and location.quality > 0:
                    return (
                        location.latitude,
                        location.longitude,
                        location.time,
                        location.satellites
                    )
            return None
        except Exception:
            return None

    @staticmethod
    def convert_to_taipei_time(utc_time):
        """Convert UTC time to Asia/Taipei timezone.
        Args:
            utc_time: datetime.time or datetime.datetime object
        Returns:
            datetime.time object in Asia/Taipei timezone
        """
        taipei_tz = pytz.timezone('Asia/Taipei')
        
        # If input is time object, convert to datetime first
        if isinstance(utc_time, time):
            today = datetime.now().date()
            utc_time = datetime.combine(today, utc_time)
        
        # Convert to Taipei timezone
        local_dt = utc_time.replace(tzinfo=pytz.UTC).astimezone(taipei_tz)
        
        # Return time component only
        return local_dt.time()

    @staticmethod
    def calculate_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates using Haversine formula.
        Args:
            coord1: (latitude, longitude) of first point
            coord2: (latitude, longitude) of second point
        Returns:
            Distance in meters
        """
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        R = 6371000  # Earth's radius in meters

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (math.sin(delta_phi/2) * math.sin(delta_phi/2) +
             math.cos(phi1) * math.cos(phi2) *
             math.sin(delta_lambda/2) * math.sin(delta_lambda/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

def main():
    gps = GPSReader(system='GNSS').connect()
    try:
        coords = gps.get_coordinates()
        if coords:
            lat, lon, gps_time, sats = coords
            local_time = GPSReader.convert_to_taipei_time(gps_time)
            print(f"Latitude: {lat:.6f}, Longitude: {lon:.6f}")
            print(f"UTC Time: {gps_time}")
            print(f"Local Time (Taipei): {local_time}")
            print(f"Satellites: {sats}")
        else:
            print("Could not get GPS fix")
    finally:
        gps.disconnect()


if __name__ == "__main__":
    main()
