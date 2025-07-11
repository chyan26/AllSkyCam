# AllSkyCam

An advanced all-sky camera system for astronomical observation and Sun tracking with comprehensive analysis capabilities.

## Overview

The AllSkyCam system provides:
- **High-precision all-sky imaging** with fisheye lens coverage
- **Automated Sun detection and tracking** with azimuth calculation  
- **Real-time GPS integration** for accurate positioning
- **Comprehensive image analysis** including edge detection and star tracking
- **Multi-format data output** (FITS, JPEG) with rich metadata
- **Exposure sequence automation** with adaptive timing
- **Real-time visualization** and heading display

## Hardware Components

### Camera System
- **IDS Peak Compatible Camera** with wide dynamic range
- **Fisheye Lens**: [M12 Fisheye Lens](https://commonlands.com/products/m12-fisheye-lens-cil220?variant=39948508430454)
- **Field of View**: ~1065 pixel radius (full sky coverage)
- **Output Format**: 12-bit Mono, FITS and JPEG formats

### GPS Module
- **u-blox NEO-8M GPS** module for precise positioning
- **Real-time coordinate tracking** with configurable update rates
- **Heading calculation** from GPS movement data
- **Documentation**: [GY-NEO8M Technical Manual](doc/GY-NEO8M/)

### Optional IMU Integration
- **ADIS16547 Datasheet**: [Available for future integration](doc/ADIS16545-16547_DataSheet_Rev_A.pdf)

## Key Features

### Advanced Camera Control (`allskyController.py`)
- **Professional-grade camera interface** with IDS Peak SDK integration
- **Adaptive exposure control** with automatic parameter adjustment
- **Multi-threaded acquisition** for real-time processing
- **Comprehensive error handling** and recovery mechanisms
- **Thread-safe operations** with state management
- **Hot pixel correction** and image enhancement

### Simple Camera Interface (`simpleExposure.py`)
- **Streamlined camera operation** for basic image acquisition
- **Configurable batch processing** with custom parameters
- **Direct FITS output** with astronomical metadata
- **Lightweight operation** for resource-constrained environments

### Sun Detection and Analysis (`detectSun.py`)
- **Advanced sun detection algorithms** using SEP (Source Extractor Python)
- **Horizon edge detection** with HoughCircles algorithm
- **Azimuth and altitude calculation** with astronomical precision
- **Image processing pipeline** with OpenCV integration
- **Solar position prediction** using Skyfield ephemeris

### GPS Integration (`gpsHandler.py`, `ubloxReader.py`)
- **Real-time GPS tracking** with u-blox protocol support
- **Configurable update frequencies** (1-10 Hz)
- **Heading calculation** from GPS movement vectors
- **Thread-safe coordinate access** with callback support
- **Distance and bearing calculations** using Haversine formula

### Data Analysis and Visualization
- **Exposure Analysis** (`exposureAnalysis.py`): Log parsing and performance analysis
- **Heading Visualizer** (`headingVisualizer.py`): Real-time compass and GPS display
- **Algorithm Testing** (`algorithmTest.ipynb`): Interactive development notebook
- **ASRD Analysis** (`asrdAnalysis.py`): Advanced sun tracking analysis

## Algorithm

The system processes images to determine the measured azimuth of the Sun as observed by the AllSkyCam. The measured azimuth (A_m) incorporates multiple correction factors:

**A_m = A_s + H + Δ**

Where:
- **A_m**: Measured azimuth (degrees)
- **A_s**: Actual azimuth of the Sun (degrees) - calculated using astronomical data
- **H**: Flight/platform heading (degrees) - from GPS or IMU
- **Δ**: Image offset (degrees) - camera alignment and calibration offset

### Key Measurements
- All angles measured clockwise from true north
- Sub-degree accuracy with proper calibration
- Temperature compensation for long-term stability
- Real-time correction based on GPS heading

## Project Structure

### Core Modules
| Module | Description |
|--------|-------------|
| `allskyController.py` | Advanced camera control with real-time analysis |
| `simpleExposure.py` | Basic camera interface for simple acquisitions |
| `detectSun.py` | Sun detection and astronomical calculations |
| `gpsHandler.py` | GPS data processing and coordinate management |
| `ubloxReader.py` | u-blox GPS protocol implementation |

### Analysis Tools
| Module | Description |
|--------|-------------|
| `exposureAnalysis.py` | Performance analysis and log processing |
| `asrdAnalysis.py` | Advanced sun tracking analysis |
| `headingVisualizer.py` | Real-time visualization and GUI |
| `algorithmTest.ipynb` | Interactive algorithm development |

### Utilities
| Module | Description |
|--------|-------------|
| `utils.py` | Common utility functions and calculations |
| `logger_config.py` | Centralized logging configuration |
| `getPosition.py` | Position calculation utilities |
| `powerTest.py` | System power and performance testing |

### Data Files
| Directory/File | Description |
|----------------|-------------|
| `images/` | Sample images and test data |
| `logs/` | System operation logs |
| `doc/` | Hardware documentation and datasheets |
| `*.csv` | GPS data and sun position records |

## Installation

### Prerequisites
- **Python 3.8+**
- **IDS Peak SDK** (from IDS Imaging website)
- **u-blox GPS module** (optional, for real-time positioning)

### Python Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy` - Numerical computing and array processing
- `astropy` - Astronomical calculations and FITS file handling
- `opencv-python` - Image processing and computer vision
- `ids-peak` - IDS camera interface
- `pyserial` - Serial communication for GPS
- `matplotlib` - Data visualization and plotting
- `skyfield` - Astronomical calculations and ephemeris
- `scipy` - Scientific computing and optimization
- `sep` - Source Extractor Python for astronomical image analysis
- `PIL` - Python Imaging Library
- `pandas` - Data analysis and manipulation

## Usage Examples

### Basic Image Acquisition
```bash
# Simple batch acquisition
python simpleExposure.py --exposure 100 --images 10 --output basic_test

# Advanced acquisition with analysis
python allskyController.py --exposure 50 --images 20 --perform_analysis --output advanced_test
```

### Real-time Operation
```bash
# Continuous acquisition with GPS and visualization
python allskyController.py --images 0 --sleep 5 --perform_analysis --gps_update_frequency 10
```

### Analysis and Testing
```python
# Interactive analysis in Jupyter
jupyter notebook algorithmTest.ipynb

# Performance analysis
python exposureAnalysis.py --log_file logs/system_20250101_120000.log --output analysis_results
```

## Configuration Options

### Camera Settings
- `--exposure`: Exposure time in milliseconds (0.01-2000)
- `--images`: Number of images to acquire (0 = continuous)
- `--sleep`: Interval between acquisitions in seconds
- `--buffers`: Number of camera buffers to allocate
- `--output`: Output directory for images and data

### Analysis Options
- `--perform_analysis`: Enable real-time sun detection and analysis
- `--hotpixel`: Enable hot pixel correction
- `--default_lat/lon`: Default GPS coordinates if GPS unavailable

### GPS Settings
- `--gps_update_frequency`: GPS update rate in Hz (1-10)
- `--log_gps_updates`: Enable GPS logging to system log

## Data Output

### Image Files
- **FITS Format**: 16-bit astronomical standard with comprehensive metadata
- **JPEG Format**: 8-bit enhanced images with detection overlays
- **Metadata**: GPS coordinates, timestamps, exposure settings, analysis results

### Log Files
- **System logs**: Comprehensive operation logging with timestamps
- **GPS logs**: Real-time position and heading data
- **Analysis logs**: Sun detection results and astronomical calculations

### Analysis Data
- **CSV files**: Structured data for post-processing
- **JSON metadata**: Machine-readable acquisition parameters
- **Performance metrics**: Timing and accuracy statistics

## Development

### Adding New Features
1. Follow the existing logging patterns using `logger_config.py`
2. Use thread-safe operations for multi-threaded components
3. Add comprehensive error handling for hardware interfaces
4. Include unit tests for new algorithms

### Testing
- Use `algorithmTest.ipynb` for interactive algorithm development
- Test camera operations with `simpleExposure.py` before using advanced features
- Validate GPS functionality with `powerTest.py`

## Hardware Setup

### Camera Connection
1. Install IDS Peak SDK from manufacturer
2. Connect camera via USB 3.0 for optimal performance
3. Verify camera detection with IDS Peak Cockpit software

### GPS Module Setup
1. Connect u-blox NEO-8M via USB or serial
2. Configure baud rate (default: 9600)
3. Verify GPS signal reception before operation

### System Requirements
- **OS**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.14+
- **RAM**: 8GB+ recommended for image processing
- **Storage**: SSD recommended for high-frequency acquisition
- **USB**: USB 3.0 ports for camera and GPS

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For technical support:
1. Check the hardware documentation in `doc/` directory
2. Review system logs for error diagnostics
3. Test with simple examples before advanced usage
4. Ensure all dependencies are properly installed