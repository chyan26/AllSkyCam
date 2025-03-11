# AllSkyCam

Note
The region of sky is ~1065 in radius.

# AllSkyCam Camera Acquisition Script

This script is designed to control an IDS Peak camera, acquire images, and optionally perform analysis on the acquired images. The images are saved in FITS format with relevant metadata.

## Algorithm

The script processes images to determine the measured azimuth of the Sun as observed by the AllSkyCam. The measured azimuth (\(A_m\)) is a composite of three components:
- **Actual Azimuth of the Sun (\(A_s\))**: The true azimuthal position of the Sun in the sky, calculated using astronomical data.
- **Flight Heading (\(H\))**: The orientation of the flight path relative to true north.
- **Image Offset (\(\Delta\))**: An angular offset introduced by the camera's alignment or image processing.

The relationship between these components is given by the following formula:

\[ A_m = A_s + H + \Delta \]

Where:
- \(A_m\): Measured azimuth (degrees)
- \(A_s\): Actual azimuth of the Sun (degrees)
- \(H\): Flight heading (degrees)
- \(\Delta\): Image offset (degrees)

### Notes
- All angles are in degrees and measured clockwise from true north.
- The image offset (\(\Delta\)) may vary depending on camera calibration and mounting.

For a rendered version of the equation, you can use an external LaTeX renderer like Codecogs:
![Azimuth Formula](https://latex.codecogs.com/png.latex?A_m%20=%20A_s%20+%20H%20+%20%5CDelta)

## Features

- Initialize and configure IDS Peak camera
- Acquire a configurable number of images
- Save images in FITS format with metadata
- Perform analysis on the images (optional)
- Log acquisition and processing details

## Requirements

- Python 3.x
- IDS Peak SDK
- Required Python packages:
  - `argparse`
  - `logging`
  - `numpy`
  - `astropy`
  - `os`
  - `datetime`
  - `time`
  - `cv2`
  - `sep`
  - `skyfield`
  - `scipy`
  - `PIL`

## Installation

1. Install the required Python packages:
   ```sh
   pip install numpy astropy opencv-python sep skyfield scipy pillow
   ```

2. Ensure that the IDS Peak SDK is installed and properly configured on your system.

## Options
--exposure: Exposure time in milliseconds (default: 0.02)
--images: Number of images to acquire (default: 50)
--sleep: Time in seconds between exposures (default: 5)
--buffers: Number of buffers to allocate (default: None)
--output: Directory to save FITS files (default: "output")
--perform_analysis: Set this flag to perform analysis on the images
Example
Logging
The script uses the logging module to log acquisition and processing details. Logs are printed to the console.


## Image Processing Functions
### Basic Image Operations
- **cv2.inRange()**
  - Used for creating a binary mask based on pixel value thresholds
- **cv2.bitwise_and()**
  - Performs bitwise AND operation between two arrays
- **cv2.convertScaleAbs()**
  - Scales, calculates absolute values, and converts to 8-bit

### Edge Detection Functions
- **cv2.Canny()**
  - Performs edge detection using the Canny algorithm
- **cv2.dilate()**
  - Dilates an image using a specific kernel

### Circle Detection
- **cv2.HoughCircles()**
  - Detects circles in an image using the Hough Circle Transform
  
#### Parameters used:
- Method: `cv2.HOUGH_GRADIENT`
- dp = 1
- minDist = image_height/2
- param1 = 10
- param2 = 20
- minRadius = 800-1000
- maxRadius = 1200


## License
This project is licensed under the MIT License.