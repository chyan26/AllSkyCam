# /home/myuser/AllSkyCam/asrdExp.py
import re
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
import os
import logging # Added for basic logging

# Setup basic logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_timestamp(timestamp_str):
    """Parse timestamp string which might include a fractional second."""
    try:
        # Try parsing with fractional second first (as produced by logger_config.py)
        return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        # Fallback for timestamps without fractional seconds (just in case)
        try:
            return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            logging.warning(f"Could not parse timestamp: {timestamp_str}")
            return None

def find_closest_entry(target_timestamp_str, data_dict):
    """
    Find the value in data_dict whose timestamp key is closest to target_timestamp_str.
    """
    target_time = parse_timestamp(target_timestamp_str)
    if target_time is None:
        return None

    closest_time_key = None
    min_diff = float('inf')

    for time_str, value in data_dict.items():
        time = parse_timestamp(time_str)
        if time is None:
            continue
        diff = abs((target_time - time).total_seconds())
        if diff < min_diff:
            min_diff = diff
            closest_time_key = time_str

    # Optional: Add a threshold for maximum allowed difference?
    # max_allowed_diff = 1.0 # seconds
    # if min_diff > max_allowed_diff:
    #     logging.debug(f"No close entry found for {target_timestamp_str} within {max_allowed_diff}s. Min diff was {min_diff:.2f}s.")
    #     return None

    return data_dict.get(closest_time_key) if closest_time_key else None

def extract_gps_and_heading(log_file):
    # Regular expressions with updated timestamp format (YYYY-MM-DD HH:MM:SS.s)
    # Note: \. matches the literal dot, \d matches one digit
    timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d)' # Capture group 1: Timestamp

    # Match the specific log message for GPS updates
    gps_pattern = re.compile(
        timestamp_pattern + r' - INFO - idsExposure.py - Updated GPS location: Latitude=([\d.-]+), Longitude=([\d.-]+), Time=.*'
    )
    # Match the specific log message for heading difference calculation
    heading_pattern = re.compile(
        timestamp_pattern + r' - INFO - idsExposure.py - Heading difference = ([\d.-]+)'
    )
    # Match the specific log message for saving FITS files (to get sequence number)
    sequence_pattern = re.compile(
        timestamp_pattern + r' - INFO - idsExposure.py - Saved FITS file: .*_(\d{3})\.fits' # Assuming 2-digit sequence number based on idsExposure.py {exposure_num:02d}
    )
    # --- Alternative sequence pattern if it might be more digits ---
    # sequence_pattern = re.compile(
    #     timestamp_pattern + r' - INFO - idsExposure - Saved FITS file: .*_(\d+)\.fits'
    # )

    coordinates = []  # List to store (timestamp_str, lat, lon) tuples
    heading_dict = {}  # Dictionary to store timestamp_str: heading pairs
    sequence_dict = {} # Dictionary to store timestamp_str: sequence number pairs

    gps_matches = 0
    heading_matches = 0
    sequence_matches = 0

    logging.info(f"Reading log file: {log_file}")
    with open(log_file, 'r') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue

            # Check for GPS coordinates
            gps_match = gps_pattern.search(line)
            if gps_match:
                timestamp = gps_match.group(1)
                lat = float(gps_match.group(2))
                lon = float(gps_match.group(3))
                coordinates.append((timestamp, lat, lon))
                gps_matches += 1
                continue # Move to next line once matched

            # Check for heading
            heading_match = heading_pattern.search(line)
            if heading_match:
                timestamp = heading_match.group(1)
                heading = float(heading_match.group(2))
                heading_dict[timestamp] = heading
                heading_matches += 1
                continue # Move to next line

            # Check for sequence number
            sequence_match = sequence_pattern.search(line)
            if sequence_match:
                timestamp = sequence_match.group(1)
                sequence = int(sequence_match.group(2))
                sequence_dict[timestamp] = sequence
                sequence_matches += 1
                continue # Move to next line

    logging.info(f"Found {gps_matches} GPS entries.")
    logging.info(f"Found {heading_matches} Heading entries.")
    logging.info(f"Found {sequence_matches} Sequence entries.")

    if not coordinates:
        logging.warning("No GPS coordinates found. Check log file and GPS regex pattern.")
        return []

    # Combine GPS coordinates with their corresponding closest headings and sequence numbers
    result = []
    missing_headings = 0
    missing_sequences = 0
    for timestamp, lat, lon in coordinates:
        heading = find_closest_entry(timestamp, heading_dict)
        sequence = find_closest_entry(timestamp, sequence_dict)

        if heading is None:
            missing_headings += 1
        if sequence is None:
            missing_sequences += 1

        result.append((lat, lon, heading, sequence)) # Append even if None, filter later

    logging.info(f"Could not find close heading for {missing_headings}/{len(coordinates)} GPS points.")
    logging.info(f"Could not find close sequence for {missing_sequences}/{len(coordinates)} GPS points.")

    return result

def calculate_track_angle(lat1, lon1, lat2, lon2):
    """
    Calculate bearing angle between two GPS points.
    Returns None if points are the same (stationary).
    """
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dLon = lon2 - lon1
    y = np.sin(dLon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)

    # Check if points are the same (within small threshold)
    # Use a threshold appropriate for GPS precision (e.g., 1e-9 radians is ~0.6mm)
    if abs(dLon) < 1e-9 and abs(lat2 - lat1) < 1e-9:
       return None

    angle = np.degrees(np.arctan2(y, x))
    return (angle + 360) % 360 # Normalize to 0-360

def plot_coordinates(data):
    if not data:
        logging.warning("No data provided for plotting coordinates.")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[3, 1]) # Increased figure size and ratio

    # Unzip the data
    lats, lons, headings, sequences = zip(*data)

    # Convert lists to numpy arrays for easier slicing and math
    lats = np.array(lats, dtype=float)
    lons = np.array(lons, dtype=float)
    # Handle potential None values in headings before converting to float
    headings = np.array([h if h is not None else np.nan for h in headings], dtype=float)
    sequences = np.array([s if s is not None else np.nan for s in sequences], dtype=float) # Keep as float for NaN

    # Apply heading offset if needed (e.g., +100 degrees as in original)
    heading_offset = 0
    headings_adjusted = headings + heading_offset # NaN values will propagate

    # Top subplot - GPS track and vectors
    ax1.plot(lons, lats, 'b-', alpha=0.5, linewidth=1, label='GPS Track')

    # Annotate unique sequence numbers (handle NaN)
    last_sequence = -1 # Use a value that won't match NaN
    annotated_indices = set()
    for i, seq in enumerate(sequences):
        if not np.isnan(seq) and int(seq) != last_sequence:
            # Avoid overlapping annotations too closely
            can_annotate = True
            for ann_idx in annotated_indices:
                dist_sq = (lons[i] - lons[ann_idx])**2 + (lats[i] - lats[ann_idx])**2
                if dist_sq < 1e-10: # Adjust threshold based on coordinate scale
                    can_annotate = False
                    break
            if can_annotate:
                ax1.text(lons[i], lats[i], str(int(seq)), fontsize=8, color='black', alpha=0.7,
                         ha='left', va='bottom')
                last_sequence = int(seq)
                annotated_indices.add(i)


    # Calculate arrow components using heading angles
    arrow_length_factor = 0.05 # Adjust based on coordinate range
    lon_range = np.nanmax(lons) - np.nanmin(lons) if len(lons) > 1 else 0.001
    lat_range = np.nanmax(lats) - np.nanmin(lats) if len(lats) > 1 else 0.001
    arrow_length = min(lon_range, lat_range) * arrow_length_factor if min(lon_range, lat_range) > 0 else 0.0001

    step = 5 # Plot vectors every 'step' points

    # --- Heading vectors (red) ---
    valid_heading_indices = np.where(~np.isnan(headings_adjusted))[0]
    plot_heading_indices = valid_heading_indices[::step]

    if len(plot_heading_indices) > 0:
        dx_heading = arrow_length * np.sin(np.radians(headings_adjusted[plot_heading_indices]))
        dy_heading = arrow_length * np.cos(np.radians(headings_adjusted[plot_heading_indices]))
        ax1.quiver(lons[plot_heading_indices], lats[plot_heading_indices], dx_heading, dy_heading,
                  color='red', scale_units='xy', angles='xy', scale=1, width=0.003, # Use scale=1 with angles='xy'
                  headwidth=4, headlength=5, headaxislength=4.5,
                  label=f'Heading Vector (+{heading_offset:.0f} deg)')

    # --- Track vectors (green) ---
    track_angles = []
    valid_track_indices = []
    angle_differences = []

    for i in range(0, len(lats)-step, step):
        # Ensure heading exists for this point to calculate difference
        if np.isnan(headings_adjusted[i]):
            continue

        angle = calculate_track_angle(lats[i], lons[i], lats[i+step], lons[i+step])
        if angle is not None:
            track_angles.append(angle)
            valid_track_indices.append(i)
            # Calculate angle difference (Track - Heading)
            heading_angle = headings_adjusted[i]
            diff = (angle - heading_angle + 180) % 360 - 180  # Normalize to [-180, 180]
            angle_differences.append(diff)

    if track_angles:
        track_angles = np.array(track_angles)
        dx_track = arrow_length * np.sin(np.radians(track_angles))
        dy_track = arrow_length * np.cos(np.radians(track_angles))

        ax1.quiver(lons[valid_track_indices], lats[valid_track_indices], dx_track, dy_track,
                  color='green', scale_units='xy', angles='xy', scale=1, width=0.003, # Use scale=1 with angles='xy'
                  headwidth=4, headlength=5, headaxislength=4.5,
                  label='Track Vector')

    # Plot start and end points
    ax1.plot(lons[0], lats[0], 'go', markersize=8, label='Start')
    ax1.plot(lons[-1], lats[-1], 'ro', markersize=8, label='End')

    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('GPS Track with Heading and Track Vectors')
    ax1.grid(True)
    ax1.legend(fontsize='small')
    ax1.ticklabel_format(useOffset=False, style='plain') # Prevent scientific notation

    # Bottom subplot - Angle differences
    if angle_differences:
        ax2.plot(valid_track_indices, angle_differences, 'b.-', label='Angle Difference (Track - Heading)')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel(f'Point Index (every {step} points)')
        ax2.set_ylabel('Angle Difference (degrees)')
        ax2.set_title('Track Angle vs Adjusted Heading Angle Difference')
        ax2.grid(True)
        ax2.legend(fontsize='small')
        ax2.set_ylim(-190, 190) # Ensure full range is visible

        # Print statistics
        mean_diff = np.mean(angle_differences)
        std_diff = np.std(angle_differences)
        median_diff = np.median(angle_differences)
        ax2.text(0.02, 0.95,
                f'Mean diff: {mean_diff:.1f}°\nMedian diff: {median_diff:.1f}°\nStd dev: {std_diff:.1f}°',
                transform=ax2.transAxes, fontsize='small', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    else:
         ax2.text(0.5, 0.5, 'No angle differences calculated\n(Need consecutive points with valid headings)',
                  ha='center', va='center', fontsize='medium', color='grey')


    plt.tight_layout(pad=1.5) # Add padding
    plt.show()

def plot_angle_comparison(data):
    if not data:
        logging.warning("No data provided for plotting angle comparison.")
        return

    # Unzip the data
    lats, lons, headings, sequences = zip(*data)
    lats = np.array(lats, dtype=float)
    lons = np.array(lons, dtype=float)
    headings = np.array([h if h is not None else np.nan for h in headings], dtype=float)

    # Apply heading offset
    heading_offset = 100.0
    headings_adjusted = headings + heading_offset

    # Calculate track angles and differences
    track_angles = []
    indices = []
    angle_diffs = []
    valid_lats = []
    valid_lons = []

    step = 5 # Use same step as plot_coordinates for consistency
    for i in range(0, len(lats)-step, step):
        if np.isnan(headings_adjusted[i]):
            continue

        track_angle = calculate_track_angle(lats[i], lons[i], lats[i+step], lons[i+step])
        if track_angle is not None:
            track_angles.append(track_angle)
            indices.append(i)
            heading_angle = headings_adjusted[i]
            diff = (track_angle - heading_angle + 180) % 360 - 180
            angle_diffs.append(diff)
            valid_lats.append(lats[i])
            valid_lons.append(lons[i])

    if not angle_diffs:
        logging.warning("Could not calculate any angle differences for comparison plot.")
        plt.figure(figsize=(15, 6))
        plt.text(0.5, 0.5, 'No angle differences calculated', ha='center', va='center', fontsize='large', color='grey')
        plt.title('Angle Comparison Plot (No Data)')
        plt.show()
        return

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7)) # Increased figure size

    # Left subplot - GPS track with color-coded differences
    scatter = ax1.scatter(valid_lons, valid_lats,
                         c=angle_diffs,
                         cmap='RdYlBu', # Red-Yellow-Blue colormap good for divergences
                         s=50,         # Marker size
                         vmin=-180,
                         vmax=180,
                         edgecolors='k', # Add black edge for visibility
                         linewidths=0.5)
    ax1.plot(lons, lats, 'k-', alpha=0.3, linewidth=1, label='_nolegend_') # Full track in background
    cbar = plt.colorbar(scatter, ax=ax1, label='Angle Difference (Track - Heading) [degrees]')
    cbar.ax.tick_params(labelsize='small')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('GPS Track Colored by Angle Difference')
    ax1.grid(True)
    ax1.ticklabel_format(useOffset=False, style='plain')

    # Add start and end points
    ax1.plot(lons[0], lats[0], 'go', markersize=8, label='Start', markeredgecolor='k')
    ax1.plot(lons[-1], lats[-1], 'ro', markersize=8, label='End', markeredgecolor='k')
    ax1.legend(fontsize='small')


    # Right subplot - Angle differences over time/index
    ax2.plot(indices, angle_diffs, 'b.-', markersize=4)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel(f'Point Index (every {step} points)')
    ax2.set_ylabel('Angle Difference (degrees)')
    ax2.set_title('Track vs Adjusted Heading Angle Difference')
    ax2.grid(True)
    ax2.set_ylim(-190, 190)

    # Add statistics
    mean_diff = np.mean(angle_diffs)
    std_diff = np.std(angle_diffs)
    median_diff = np.median(angle_diffs)
    ax2.text(0.02, 0.95,
            f'Mean diff: {mean_diff:.1f}°\nMedian diff: {median_diff:.1f}°\nStd dev: {std_diff:.1f}°',
            transform=ax2.transAxes, fontsize='small', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout(pad=1.5)
    plt.show()


def find_heading_anomalies(data, track_threshold=5, angle_difference_threshold=30):
    """
    Identify cases where the heading vector has a large angle difference
    compared to the track vector, while the track vector remains relatively constant.

    Parameters:
    - data: List of (Latitude, Longitude, Heading, Sequence) tuples
    - track_threshold: Maximum allowed variation in track vector (degrees)
    - angle_difference_threshold: Minimum required angle difference between track and heading vectors (degrees)

    Returns:
    - anomalies: List of indices where the anomaly occurs
    """
    if not data:
        return []

    lats, lons, headings, sequences = zip(*data)
    lats = np.array(lats, dtype=float)
    lons = np.array(lons, dtype=float)
    headings = np.array([h if h is not None else np.nan for h in headings], dtype=float)
    sequences = np.array([s if s is not None else np.nan for s in sequences], dtype=float)

    # Apply heading offset
    heading_offset = 100.0
    headings_adjusted = headings + heading_offset

    track_angles = []
    anomalies = []

    step = 5 # Use consistent step size
    # Need at least 2*step points to compare track variation
    if len(lats) < 2 * step:
        logging.warning("Not enough data points to calculate track variations for anomaly detection.")
        return []

    # Calculate initial track angles
    current_track_angles = {}
    for i in range(0, len(lats) - step, step):
         angle = calculate_track_angle(lats[i], lons[i], lats[i + step], lons[i + step])
         if angle is not None:
             current_track_angles[i] = angle

    # Iterate and check for anomalies
    for i in range(step, len(lats) - step, step):
        # Check if we have track angles for current and previous segments
        prev_idx = i - step
        if i not in current_track_angles or prev_idx not in current_track_angles:
            continue

        # Check if heading exists for the current point
        if np.isnan(headings_adjusted[i]):
            continue

        current_track = current_track_angles[i]
        prev_track = current_track_angles[prev_idx]

        # Calculate track vector variation (difference between consecutive track angles)
        track_variation = abs(current_track - prev_track)
        # Normalize variation angle difference to [0, 180]
        track_variation = min(track_variation, 360 - track_variation)

        # Calculate angle difference between current track and heading vectors
        heading_angle = headings_adjusted[i]
        angle_difference = (current_track - heading_angle + 180) % 360 - 180  # Normalize to [-180, 180]

        # Check for anomalies
        if track_variation < track_threshold and abs(angle_difference) > angle_difference_threshold:
            anomalies.append(i) # Report index 'i' where the anomaly condition is met

    return anomalies

def main():
    # --- Configuration ---
    log_file = 'logs/system_20250422_151314.log' # Make sure this path is correct
    csv_file = 'gps_data_extracted.csv' # Use a different name to avoid confusion
    force_reextract = False # Set to True to ignore existing CSV and re-parse log
    # --- End Configuration ---

    if not os.path.exists(log_file):
        logging.error(f"Log file not found: {log_file}")
        return

    data = []
    if os.path.exists(csv_file) and not force_reextract:
        logging.info(f"Loading existing data from {csv_file}")
        try:
            df = pd.read_csv(csv_file)
            # Handle potential empty strings read as NaN, convert relevant columns
            df['Heading'] = pd.to_numeric(df['Heading'], errors='coerce')
            df['Sequence'] = pd.to_numeric(df['Sequence'], errors='coerce')
            # Convert DataFrame back to the list of tuples format, preserving NaN where needed
            data = [
                (row['Latitude'], row['Longitude'],
                 row['Heading'] if pd.notna(row['Heading']) else None,
                 row['Sequence'] if pd.notna(row['Sequence']) else None)
                for index, row in df.iterrows()
            ]
            logging.info(f"Loaded {len(data)} records from CSV.")
        except Exception as e:
            logging.error(f"Error loading data from {csv_file}: {e}. Re-extracting from log.")
            data = [] # Ensure data is empty to trigger extraction

    if not data: # If CSV didn't exist, was empty, force_reextract, or loading failed
        logging.info(f"Extracting GPS data from {log_file}")
        raw_data = extract_gps_and_heading(log_file)

        if not raw_data:
            logging.error("No data extracted from log file. Exiting.")
            return

        # Filter out entries where essential data (lat, lon) might be missing (though unlikely from regex)
        # And convert to DataFrame for saving
        df = pd.DataFrame(raw_data, columns=['Latitude', 'Longitude', 'Heading', 'Sequence'])
        df.dropna(subset=['Latitude', 'Longitude'], inplace=True) # Should not drop any if regex worked

        # Save to CSV (includes None/NaN values correctly)
        try:
            df.to_csv(csv_file, index=False)
            logging.info(f"Saved extracted data to {csv_file}")
        except Exception as e:
            logging.error(f"Failed to save data to CSV {csv_file}: {e}")

        # Prepare data for plotting (list of tuples format)
        data = [
            (row['Latitude'], row['Longitude'],
             row['Heading'] if pd.notna(row['Heading']) else None,
             row['Sequence'] if pd.notna(row['Sequence']) else None)
            for index, row in df.iterrows()
        ]

    if not data:
        logging.error("No data available for analysis after extraction/loading. Exiting.")
        return

    logging.info(f"Total GPS coordinates for analysis: {len(data)}")
    if data:
        logging.info("\nFirst coordinate:")
        logging.info(f"Lat: {data[0][0]}, Lon: {data[0][1]}, Heading: {data[0][2]}, Seq: {data[0][3]}")
        logging.info("\nLast coordinate:")
        logging.info(f"Lat: {data[-1][0]}, Lon: {data[-1][1]}, Heading: {data[-1][2]}, Seq: {data[-1][3]}")

    # Perform plotting and analysis
    plot_coordinates(data)
    plot_angle_comparison(data)

    # Find anomalies
    anomalies = find_heading_anomalies(data)
    logging.info(f"\nNumber of potential heading anomalies found: {len(anomalies)}")
    if anomalies:
        logging.info("Anomaly Indices (relative to filtered data, using step=5):")
        for idx in anomalies:
             # Ensure index is valid before accessing data
            if 0 <= idx < len(data):
                logging.info(f"  Index {idx}: Lat={data[idx][0]:.6f}, Lon={data[idx][1]:.6f}, Hdg={data[idx][2]}, Seq={data[idx][3]}")
            else:
                logging.warning(f"  Anomaly index {idx} out of bounds for data length {len(data)}")


    # Display DataFrame summary if loaded/created
    if 'df' in locals() and not df.empty:
        logging.info("\nDataFrame Summary:")
        # Configure pandas display options for better readability
        pd.set_option('display.float_format', '{:.6f}'.format)
        print(df.describe(include='all')) # Include non-numeric columns too
    else:
        logging.info("\nNo DataFrame available for summary.")


if __name__ == "__main__":
    main()