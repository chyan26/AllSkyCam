#!/usr/bin/env python3
"""
Example usage of the unified CameraAcquisition class
"""

import argparse
import logging
import time
from unified_camera import CameraAcquisition, create_simple_camera, create_advanced_camera

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s:%(lineno)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)

logger = logging.getLogger(__name__)


def example_simple_batch():
    """Example: Simple batch image acquisition (like simpleExposure)."""
    logger.info("=== Simple Batch Mode Example ===")
    
    # Create camera with simple configuration
    camera = create_simple_camera(
        exposure_time_ms=100.0,  # 100ms exposure
        output_dir="simple_output",
        hotpixel_correction=True
    )
    
    try:
        # Initialize camera
        if not camera.initialize():
            logger.error("Camera initialization failed")
            return False
        
        # Capture batch of images
        num_images = 5
        success = camera.process_batch_images(num_images)
        
        if success:
            logger.info(f"Successfully captured {num_images} images")
        else:
            logger.error("Batch capture failed")
            
        return success
        
    finally:
        camera.cleanup()


def example_advanced_continuous():
    """Example: Advanced continuous acquisition (like allskyController)."""
    logger.info("=== Advanced Continuous Mode Example ===")
    
    # Create camera with advanced configuration
    camera = create_advanced_camera(
        exposure_time_ms=20.0,  # 20ms exposure
        output_dir="advanced_output",
        enable_jpeg_output=True
    )
    
    try:
        # Initialize camera
        if not camera.initialize():
            logger.error("Camera initialization failed")
            return False
        
        # Start continuous acquisition
        if not camera.start_acquisition():
            logger.error("Failed to start acquisition")
            return False
        
        # Simulate GPS data update
        camera.update_gps_coordinates(25.0330, 121.5654)  # Taipei coordinates
        
        # Acquire images continuously
        for i in range(10):
            logger.info(f"Acquiring image {i+1}/10...")
            
            image_data, buffer = camera.acquire_single_image(timeout_ms=3000)
            
            if image_data is not None:
                # Simulate analysis results
                camera.update_analysis_results(
                    measured_alt=45.0 + i,  # Simulated sun altitude
                    measured_azi=180.0 + i*2,  # Simulated sun azimuth
                    head_diff=i*0.1  # Simulated heading difference
                )
                
                # Save with metadata
                if camera.save_fits(image_data, i+1):
                    logger.info(f"Saved FITS for image {i+1}")
                    
                if camera.save_jpeg(image_data, i+1):
                    logger.info(f"Saved JPEG for image {i+1}")
                
                # Queue buffer back
                if buffer:
                    camera.queue_buffer(buffer)
                    
            else:
                logger.warning(f"Failed to acquire image {i+1}")
            
            # Small delay between images
            time.sleep(0.1)
        
        return True
        
    finally:
        camera.cleanup()


def example_context_manager():
    """Example: Using camera with context manager for automatic cleanup."""
    logger.info("=== Context Manager Example ===")
    
    try:
        with CameraAcquisition(
            exposure_time_ms=50.0,
            output_dir="context_output",
            enable_jpeg_output=True
        ) as camera:
            
            # Camera is automatically initialized
            logger.info("Camera initialized via context manager")
            
            # Process a few images
            camera.process_batch_images(3)
            
            logger.info("Batch processing complete")
            
        # Camera is automatically cleaned up when exiting context
        logger.info("Camera cleaned up automatically")
        return True
        
    except Exception as e:
        logger.error(f"Error in context manager example: {e}")
        return False


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Unified Camera Acquisition Examples")
    parser.add_argument("--mode", choices=["simple", "advanced", "context"], 
                       default="simple", help="Example mode to run")
    parser.add_argument("--exposure", type=float, default=100.0, 
                       help="Exposure time in milliseconds")
    parser.add_argument("--output", type=str, default="example_output",
                       help="Output directory")
    
    args = parser.parse_args()
    
    logger.info(f"Starting unified camera example in {args.mode} mode")
    
    if args.mode == "simple":
        success = example_simple_batch()
    elif args.mode == "advanced":
        success = example_advanced_continuous()
    elif args.mode == "context":
        success = example_context_manager()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1
    
    if success:
        logger.info("Example completed successfully")
        return 0
    else:
        logger.error("Example failed")
        return 1


if __name__ == "__main__":
    exit(main())
