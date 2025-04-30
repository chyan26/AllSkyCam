from datetime import time, datetime
from typing import Tuple
import pytz
import math

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
    return local_dt

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