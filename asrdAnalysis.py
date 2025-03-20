import os
import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from datetime import datetime
from detectSun import ImageProcessor
from headingVisualizer import HeadingVisualizer
import tkinter as tk

def extract_lat_lon_azi_from_fits(file_path):
    with fits.open(file_path) as hdul:
        header = hdul[0].header
        latitude = header.get('LATITUDE', None)  # Use get() to avoid KeyError
        longitude = header.get('LONGITUD', None)
        azimuth = header.get('MEAS_AZI', 0)
    return latitude, longitude, azimuth

def parse_datetime_from_filename(filename):
    basename = os.path.basename(filename)
    try:
        date_str = basename.split('_')[1] + basename.split('_')[2] + basename.split('_')[3] + basename.split('_')[4]
        return datetime.strptime(date_str, '%Y%m%d%H%M%S')
    except (IndexError, ValueError) as e:
        print(f"Error parsing date from filename {filename}: {e}")
        return None

def collect_lat_lon_data(file_pattern, start_time, end_time):
    latitudes = []
    longitudes = []
    azimuths = []
    labels = []
    fits_files = glob.glob(file_pattern)
    fits_files.sort()
    print(f"Found {len(fits_files)} files matching the pattern.")
    
    for fits_file in fits_files:
        file_datetime = parse_datetime_from_filename(fits_file)
        if file_datetime is None:
            continue
        if start_time <= file_datetime <= end_time:
            lat, lon, azi = extract_lat_lon_azi_from_fits(fits_file)
            print(f"File {fits_file} has LATITUDE={lat}, LONGITUDE={lon}, AZIMUTH={azi}")
            if lat is not None and lon is not None:
                latitudes.append(lat)
                longitudes.append(lon)
                azimuths.append(azi)
                label = os.path.basename(fits_file).split('_')[-1].split('.')[0]
                labels.append(label)
            else:
                print(f"File {fits_file} does not contain LATITUDE or LONGITUDE in the header.")
    
    return latitudes, longitudes, azimuths, labels

def calculate_rotation_angle(measured_azimuth, actual_azimuth):
    rotation_angle = actual_azimuth - measured_azimuth
    return rotation_angle

def plot_lat_lon(latitudes, longitudes, labels, solar_azimuths):
    if not latitudes or not longitudes:
        print("No data to plot.")
        return
    
    plt.scatter(longitudes, latitudes, marker='o', color='b')
    
    lon_range = max(longitudes) - min(longitudes)
    lat_range = max(latitudes) - min(latitudes)
    arrow_length = min(lon_range, lat_range) * 0.005
    
    for i, label in enumerate(labels):
        plt.text(longitudes[i], latitudes[i], label, fontsize=9, ha='right')
    
    u = []
    v = []
    for i in range(len(latitudes) - 1):
        delta_lon = longitudes[i + 1] - longitudes[i]
        delta_lat = latitudes[i + 1] - latitudes[i]
        norm = np.sqrt(delta_lon**2 + delta_lat**2)
        if norm != 0:
            u.append(arrow_length * (delta_lon / norm))
            v.append(arrow_length * (delta_lat / norm))
        else:
            u.append(0)
            v.append(0)
    
    u.append(0)
    v.append(0)
    
    solar_azimuths = 90 - np.array(solar_azimuths) % 360
    solar_u = [arrow_length * np.cos(np.radians(azi)) for azi in solar_azimuths]
    solar_v = [arrow_length * np.sin(np.radians(azi)) for azi in solar_azimuths]
    
    plt.quiver(longitudes, latitudes, solar_u, solar_v, angles='xy', scale_units='xy', scale=0.15, color='g')
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Latitude vs Longitude with Direction Vectors and Solar Azimuths')
    plt.grid(True)
    plt.show()

def runAnalysis():
    file_pattern = os.path.expanduser("~/AllSkyCam/output/image_*.fits")
    start_time = datetime(2025, 3, 11, 15, 42, 41)
    end_time = datetime(2025, 3, 11, 15, 47, 54)

    fits_files = glob.glob(file_pattern)
    fits_files.sort()

    filtered_files = [
        fits_file for fits_file in fits_files
        if start_time <= parse_datetime_from_filename(fits_file) <= end_time
    ]

    print(f"Found {len(filtered_files)} FITS files in the specified time range.")
    
    if not filtered_files:
        print("No files to process.")
        return

    # Collect data for plotting
    latitudes, longitudes, azimuths, labels = collect_lat_lon_data(file_pattern, start_time, end_time)

    # Initialize tkinter root and visualizer
    root = tk.Tk()
    visualizer = HeadingVisualizer(root)
    
    # Process files one by one with real-time updates
    def process_files(i=0):
        if i < len(filtered_files):
            fits_file = filtered_files[i]
            
            # Process the current file
            with fits.open(fits_file) as hdul:
                header = hdul[0].header
                heading = header.get('HEAD_DIF', 0)
                
                # Update the visualizer
                visualizer.update_heading(heading)
                
                # Display filename
                visualizer.canvas.delete("filename_text")
                visualizer.canvas.create_text(150, 270, text=os.path.basename(fits_file), tags="filename_text")
                
                print(f"File: {os.path.basename(fits_file)}, Heading: {heading}")
            
            # Schedule next file
            root.after(1000, lambda: process_files(i+1))
        else:
            print("All files processed.")
            # Plot the latitude/longitude data after processing
            plot_lat_lon(latitudes, longitudes, labels, azimuths)
    
    # Start processing files
    process_files()
    
    # Start the tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    runAnalysis()