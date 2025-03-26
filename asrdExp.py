import re
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

def parse_timestamp(timestamp_str):
    return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

def find_closest_heading(timestamp_str, heading_dict):
    target_time = parse_timestamp(timestamp_str)
    closest_time = None
    min_diff = float('inf')
    
    for time_str in heading_dict:
        time = parse_timestamp(time_str)
        diff = abs((target_time - time).total_seconds())
        if diff < min_diff:
            min_diff = diff
            closest_time = time_str
    
    return heading_dict.get(closest_time) if closest_time else None

def extract_gps_and_heading(log_file):
    # Regular expressions for GPS and heading
    gps_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - INFO - idsExposure\.py - Updated GPS location: Latitude=([\d.-]+), Longitude=([\d.-]+), Time=.*'
    heading_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - INFO - idsExposure\.py - Heading difference = ([\d.-]+)'
    
    coordinates = []  # List to store (timestamp, lat, lon) tuples
    heading_dict = {}  # Dictionary to store timestamp: heading pairs
    
    with open(log_file, 'r') as file:
        for line in file:
            # Check for GPS coordinates
            gps_match = re.search(gps_pattern, line)
            if gps_match:
                timestamp = gps_match.group(1)
                lat = float(gps_match.group(2))
                lon = float(gps_match.group(3))
                coordinates.append((timestamp, lat, lon))
            
            # Check for heading
            heading_match = re.search(heading_pattern, line)
            if heading_match:
                timestamp = heading_match.group(1)
                heading = float(heading_match.group(2))
                heading_dict[timestamp] = heading
    
    # Combine GPS coordinates with their corresponding headings
    result = []
    for timestamp, lat, lon in coordinates:
        heading = find_closest_heading(timestamp, heading_dict)
        result.append((lat, lon, heading))
    
    return result

def plot_coordinates(data):
    # Unzip the data
    lats, lons, headings = zip(*data)
    
    plt.figure(figsize=(12, 8))
    
    # Convert lists to numpy arrays for easier slicing
    lats = np.array(lats)
    lons = np.array(lons)
    headings = np.array(headings)+100
    
    # Plot all GPS points as a continuous track
    plt.plot(lons, lats, 'b-', alpha=0.5, linewidth=1)
    plt.scatter(lons, lats, c='black', s=20, alpha=0.5)

    # Calculate arrow components using heading angles for every 10th point
    arrow_length = 0.0001
    dx = arrow_length * np.sin(np.radians(headings[::10]))
    dy = arrow_length * np.cos(np.radians(headings[::10]))
    
    # Add arrows for heading every 10th point
    plt.quiver(lons[::10], lats[::10], dx, dy,
              color='red',  # Single color for all arrows
              scale=0.002,
              width=0.003,
              headwidth=4,
              headlength=5,
              headaxislength=4.5)
    
    # Plot start and end points
    plt.plot(lons[0], lats[0], 'go', label='Start', markersize=10)
    plt.plot(lons[-1], lats[-1], 'ro', label='End', markersize=10)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('GPS Coordinates Track with Heading Vectors')
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    log_file = 'system_20250321_115636.log'
    data = extract_gps_and_heading(log_file)
    
    print(f"Number of GPS coordinates extracted: {len(data)}")
    print("\nFirst coordinate:")
    print(f"Latitude: {data[0][0]}, Longitude: {data[0][1]}, Heading: {data[0][2]}")
    print("\nLast coordinate:")
    print(f"Latitude: {data[-1][0]}, Longitude: {data[-1][1]}, Heading: {data[-1][2]}")
    
    plot_coordinates(data)

if __name__ == "__main__":
    main()