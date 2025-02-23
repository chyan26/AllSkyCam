import os
import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from datetime import datetime

def extract_lat_lon_azi_from_fits(file_path):
    with fits.open(file_path) as hdul:
        header = hdul[0].header
        latitude = header['LATITUDE']
        longitude = header['LONGITUD']
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

def plot_lat_lon(latitudes, longitudes, azimuths, labels):
    if not latitudes or not longitudes:
        print("No data to plot.")
        return
    
    plt.scatter(longitudes, latitudes, marker='o', color='b')
    
    # Calculate the range of the data
    lon_range = max(longitudes) - min(longitudes)
    lat_range = max(latitudes) - min(latitudes)
    
    # Set the arrow length as a fraction of the data range
    arrow_length = min(lon_range, lat_range) * 0.02  # Adjusted to 2% of the smaller range
    
    for i, label in enumerate(labels):
        plt.text(longitudes[i], latitudes[i], label, fontsize=9, ha='right')
    
    # Calculate vector components based on azimuth
    # Adjust azimuth to be measured from north and increase clockwise
    adjusted_azimuths = (90 - np.array(azimuths)) % 360
    u = arrow_length * np.cos(np.radians(adjusted_azimuths))
    v = arrow_length * np.sin(np.radians(adjusted_azimuths))
    
    plt.quiver(longitudes, latitudes, u, v, angles='xy', scale_units='xy', scale=0.3, color='r')
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Latitude vs Longitude with Azimuth Vectors')
    plt.grid(True)
    plt.show()

def runAnalysis():
    file_pattern = os.path.expanduser("~/AllSkyCam/output/image_*.fits")
    start_time = datetime(2025, 2, 11, 14, 43, 31)  # Start time: 2025-02-11 14:43:31
    end_time = datetime(2025, 2, 11, 14, 49, 14)    # End time: 2025-02-11 14:49:14
    latitudes, longitudes, azimuths, labels = collect_lat_lon_data(file_pattern, start_time, end_time)
    print(f"Latitudes: {latitudes}")
    print(f"Longitudes: {longitudes}")
    print(f"Azimuths: {azimuths}")
    print(f"Labels: {labels}")
    plot_lat_lon(latitudes, longitudes, azimuths, labels)

if __name__ == "__main__":
    runAnalysis()