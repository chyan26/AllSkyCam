import re
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
import os

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

def find_closest_sequence(timestamp_str, sequence_dict):
    """
    Find the closest sequence number for a given timestamp.
    """
    target_time = parse_timestamp(timestamp_str)
    closest_time = None
    min_diff = float('inf')
    
    for time_str in sequence_dict:
        time = parse_timestamp(time_str)
        diff = abs((target_time - time).total_seconds())
        if diff < min_diff:
            min_diff = diff
            closest_time = time_str
    
    return sequence_dict.get(closest_time) if closest_time else None
def extract_gps_and_heading(log_file):
    # Regular expressions for GPS, heading, and sequence number
    gps_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - INFO - idsExposure\.py - Updated GPS location: Latitude=([\d.-]+), Longitude=([\d.-]+), Time=.*'
    heading_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - INFO - idsExposure\.py - Heading difference = ([\d.-]+)'
    sequence_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - INFO - idsExposure\.py - Saved FITS file: .*_(\d+)\.fits'

    coordinates = []  # List to store (timestamp, lat, lon) tuples
    heading_dict = {}  # Dictionary to store timestamp: heading pairs
    sequence_dict = {}  # Dictionary to store timestamp: sequence number pairs

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
            
            # Check for sequence number
            sequence_match = re.search(sequence_pattern, line)
            if sequence_match:
                timestamp = sequence_match.group(1)
                sequence = int(sequence_match.group(2))
                sequence_dict[timestamp] = sequence

    # Combine GPS coordinates with their corresponding headings and sequence numbers
    result = []
    for timestamp, lat, lon in coordinates:
        heading = find_closest_heading(timestamp, heading_dict)
        sequence = find_closest_sequence(timestamp, sequence_dict)  # Find closest sequence number
        result.append((lat, lon, heading, sequence))
    
    return result

def plot_coordinates(data):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), height_ratios=[2, 1])
    
    # Unzip the data
    lats, lons, headings, sequences = zip(*data)
    
    # Convert lists to numpy arrays for easier slicing
    lats = np.array(lats)
    lons = np.array(lons)
    headings = np.array(headings)+100
    #sequences = np.array(sequences, dtype=int)
    
    # Top subplot - GPS track and vectors
    ax1.plot(lons, lats, 'b-', alpha=0.5, linewidth=1)

    # Annotate unique sequence numbers
    last_sequence = None
    for i, seq in enumerate(sequences):
        if seq != np.nan and seq != last_sequence:  # Only annotate if sequence is unique
            print(seq)
            ax1.text(lons[i], lats[i], str(int(seq)), fontsize=8, color='black', alpha=0.7)
            last_sequence = seq

    # Calculate arrow components using heading angles for every 10th point
    arrow_length = 0.0001
    step = 5
    
    # Heading vectors (red)
    dx_heading = arrow_length * np.sin(np.radians(headings[::step]))
    dy_heading = arrow_length * np.cos(np.radians(headings[::step]))
    
    # Track vectors (green)
    track_angles = []
    valid_indices = []
    angle_differences = []
    
    for i in range(0, len(lats)-step, step):
        angle = calculate_track_angle(lats[i], lons[i], lats[i+step], lons[i+step])
        if angle is not None:
            track_angles.append(angle)
            valid_indices.append(i)
            # Calculate angle difference
            heading_angle = headings[i]
            diff = (angle - heading_angle + 180) % 360 - 180  # Normalize to [-180, 180]
            angle_differences.append(diff)
    
    if track_angles:
        track_angles = np.array(track_angles)
        dx_track = arrow_length * np.sin(np.radians(track_angles))
        dy_track = arrow_length * np.cos(np.radians(track_angles))
        
        ax1.quiver(lons[valid_indices], lats[valid_indices], dx_track, dy_track,
                  color='green', scale=0.002, width=0.003,
                  headwidth=4, headlength=5, headaxislength=4.5,
                  label='Track Vector')
    
    # Plot heading vectors
    ax1.quiver(lons[::step], lats[::step], dx_heading, dy_heading,
              color='red', scale=0.002, width=0.003,
              headwidth=4, headlength=5, headaxislength=4.5,
              label='Heading Vector')
    
    # Plot start and end points
    ax1.plot(lons[0], lats[0], 'go', label='Start', markersize=10)
    ax1.plot(lons[-1], lats[-1], 'ro', label='End', markersize=10)
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('GPS Coordinates Track with Heading and Track Vectors')
    ax1.grid(True)
    ax1.legend()
    
    # Bottom subplot - Angle differences
    if angle_differences:
        ax2.plot(valid_indices, angle_differences, 'b.-', label='Angle Difference')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Point Index (every 10 points)')
        ax2.set_ylabel('Angle Difference (degrees)')
        ax2.set_title('Track Angle vs Heading Angle Difference')
        ax2.grid(True)
        ax2.legend()
        
        # Print statistics
        mean_diff = np.mean(angle_differences)
        std_diff = np.std(angle_differences)
        ax2.text(0.02, 0.95, 
                f'Mean diff: {mean_diff:.1f}°\nStd dev: {std_diff:.1f}°', 
                transform=ax2.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def calculate_track_angle(lat1, lon1, lat2, lon2):
    """
    Calculate bearing angle between two GPS points.
    Returns None if points are the same (stationary).
    """
    dy = lat2 - lat1
    dx = lon2 - lon1
    
    # Check if points are the same (within small threshold for floating point comparison)
    if abs(dx) < 1e-10 and abs(dy) < 1e-10:
        return None
        
    angle = np.degrees(np.arctan2(dx, dy))
    return (angle + 360) % 360

def plot_angle_comparison(data):
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Unzip the data
    lats, lons, headings, sequences = zip(*data)
    lats = np.array(lats)
    lons = np.array(lons)
    headings = np.array(headings)+100
    
    # Calculate track angles and differences
    track_angles = []
    indices = []
    angle_diffs = []
    valid_lats = []
    valid_lons = []
    
    step = 10
    for i in range(0, len(lats)-step, step):
        track_angle = calculate_track_angle(lats[i], lons[i], lats[i+step], lons[i+step])
        if track_angle is not None:
            track_angles.append(track_angle)
            indices.append(i)
            heading_angle = headings[i]
            diff = (track_angle - heading_angle + 180) % 360 - 180
            angle_diffs.append(diff)
            valid_lats.append(lats[i])
            valid_lons.append(lons[i])
    
    # Left subplot - GPS track with color-coded differences
    scatter = ax1.scatter(valid_lons, valid_lats, 
                         c=angle_diffs, 
                         cmap='RdYlBu', 
                         s=100,
                         vmin=-180, 
                         vmax=180)
    ax1.plot(lons, lats, 'k-', alpha=0.3, linewidth=1)
    plt.colorbar(scatter, ax=ax1, label='Angle Difference (degrees)')
    
    # Add start and end points
    ax1.plot(lons[0], lats[0], 'go', label='Start', markersize=10)
    ax1.plot(lons[-1], lats[-1], 'ro', label='End', markersize=10)
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('GPS Track with Angle Differences')
    ax1.grid(True)
    ax1.legend()
    
    # Right subplot - Angle differences over time
    ax2.plot(indices, angle_diffs, 'b.-')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Point Index (every 10 points)')
    ax2.set_ylabel('Angle Difference (degrees)')
    ax2.set_title('Track vs Heading Angle Difference')
    ax2.grid(True)
    
    # Add statistics
    mean_diff = np.mean(angle_diffs)
    std_diff = np.std(angle_diffs)
    ax2.text(0.02, 0.95, 
            f'Mean diff: {mean_diff:.1f}°\nStd dev: {std_diff:.1f}°',
            transform=ax2.transAxes,
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def find_heading_anomalies(data, track_threshold=5, angle_difference_threshold=30):
    """
    Identify cases where the heading vector has a large angle difference
    compared to the track vector, while the track vector remains relatively constant.
    
    Parameters:
    - data: List of (Latitude, Longitude, Heading) tuples
    - track_threshold: Maximum allowed variation in track vector (degrees)
    - angle_difference_threshold: Minimum required angle difference between track and heading vectors (degrees)
    
    Returns:
    - anomalies: List of indices where the anomaly occurs
    """
    lats, lons, headings, sequences = zip(*data)
    lats = np.array(lats)
    lons = np.array(lons)
    headings = np.array(headings)
    
    track_angles = []
    track_variations = []
    angle_differences = []
    anomalies = []
    
    step = 10
    for i in range(0, len(lats) - step, step):
        # Calculate track angle
        track_angle = calculate_track_angle(lats[i], lons[i], lats[i + step], lons[i + step])
        if track_angle is not None:
            track_angles.append(track_angle)
            
            # Calculate track vector variation
            if len(track_angles) > 1:
                track_variation = abs(track_angles[-1] - track_angles[-2])
                track_variation = (track_variation + 180) % 360 - 180  # Normalize to [-180, 180]
                track_variations.append(abs(track_variation))
                
                # Calculate angle difference between track and heading vectors
                heading_angle = headings[i]+100
                angle_difference = (track_angle - heading_angle + 180) % 360 - 180  # Normalize to [-180, 180]
                angle_differences.append(abs(angle_difference))
                
                # Check for anomalies
                if track_variation < track_threshold and abs(angle_difference) > angle_difference_threshold:
                    anomalies.append(i)
    
    return anomalies

def main():
    log_file = 'system_20250321_115636.log'
    csv_file = 'gps_data.csv'
    
    # Check if CSV file exists
    if os.path.exists(csv_file):
        print(f"Loading existing data from {csv_file}")
        df = pd.read_csv(csv_file)
        # Convert DataFrame back to the format needed for plotting
        data = list(zip(df['Latitude'], df['Longitude'], df['Heading'], df['Sequence']))
    else:
        print(f"Extracting GPS data from {log_file}")
        data = extract_gps_and_heading(log_file)
        # Save to DataFrame and CSV
        df = pd.DataFrame(data, columns=['Latitude', 'Longitude', 'Heading', 'Sequence'])
        df.to_csv(csv_file, index=False)

    print(f"Number of GPS coordinates: {len(data)}")
    print("\nFirst coordinate:")
    print(f"Latitude: {data[0][0]}, Longitude: {data[0][1]}, Heading: {data[0][2]}, Sequence: {data[0][3]}")
    print("\nLast coordinate:")
    print(f"Latitude: {data[-1][0]}, Longitude: {data[-1][1]}, Heading: {data[-1][2]}, Sequence: {data[-1][3]}")
    
    plot_coordinates(data)
    plot_angle_comparison(data)
    
    # Find anomalies
    anomalies = find_heading_anomalies(data)
    print(f"\nNumber of anomalies found: {len(anomalies)}")
    for idx in anomalies:
        print(f"Anomaly at index {idx}: Latitude={data[idx][0]}, Longitude={data[idx][1]}, Heading={data[idx][2]}, Sequence={data[idx][3]}")
    
    print("\nDataFrame Summary:")
    print(df.describe())


if __name__ == "__main__":
    main()