# AllSkyCam Dashboard Demo Programs

This directory contains two focused demo programs to showcase the AllSkyCam Dashboard capabilities.

## Demo Programs

### `allskyDashboard_basic_demo.py`
**Purpose**: Basic dashboard demonstration with sample data
**Features**:
- Loads a sample FITS image from the images directory
- Sets static GPS, IMU, and sun tracking data
- Demonstrates minimal dashboard usage with fixed values
- Perfect for understanding the basic API

**Usage**:
```bash
python allskyDashboard_basic_demo.py
```

### `allskyDashboard_external_demo.py`
**Purpose**: Advanced external data sources demonstration
**Features**:
- Mock GPS handler with realistic coordinate changes
- Mock IMU handler with simulated sensor data (roll, pitch, yaw)
- Mock sun tracker with calculated headings
- Automatic image cycling through FITS files
- Demonstrates full integration with external data sources
- Shows complete external data integration

**Usage**:
```bash
python allskyDashboard_external_demo.py
```

## Core Files

### `dashboard.py`
Main dashboard class - pure display component

### `fits_handler.py`
FITS file management utility

## Quick Start

For first-time users, start with:
```bash
python allskyDashboard_basic_demo.py
```

**Usage**:
```bash
python allskyDashboard_external_demo.py
```

## Choosing the Right Demo

- **Start with `allskyDashboard_basic_demo.py`** if you want to understand the basic dashboard API
- **Use `allskyDashboard_external_demo.py`** to see how to integrate with external data sources

## Integration with Real Hardware

Both demos show the same external API used by the dashboard. To integrate with real AllSkyCam hardware:

1. Replace the mock data sources in `allskyDashboard_external_demo.py` with your actual hardware interfaces
2. Use the same `update_*` methods shown in both demos
3. See `EXTERNAL_DATA_API.md` for detailed API documentation

## File Requirements

Both demos require FITS image files in the `images/` directory. The basic demo uses any single FITS file, while the external demo cycles through multiple files to simulate a live feed.
