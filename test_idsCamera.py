#!/usr/bin/env python3
"""
Testing script for the IDS Peak Camera Acquisition system (ids_camera.py)

This script provides comprehensive testing capabilities for the CameraAcquisition class,
including unit tests, integration tests, and performance benchmarks.

Usage:
    python test_ids_camera.py --test basic          # Basic functionality test
    python test_ids_camera.py --test batch          # Batch acquisition test  
    python test_ids_camera.py --test continuous     # Continuous acquisition test
    python test_ids_camera.py --test stress         # Stress test
    python test_ids_camera.py --test all           # Run all tests
"""

import argparse
import logging
import time
import sys
import os
import unittest
from typing import Dict, List, Tuple
import numpy as np
from idsCamera import CameraAcquisition, create_simple_camera, create_advanced_camera

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(filename)s:%(lineno)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)

logger = logging.getLogger(__name__)


class CameraTestSuite:
    """Test suite for IDS Camera functionality."""
    
    def __init__(self, output_dir: str = "test_output"):
        self.output_dir = output_dir
        self.test_results: Dict[str, bool] = {}
        self.test_times: Dict[str, float] = {}
        
        # Create test output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and record results."""
        logger.info(f"{'='*60}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        try:
            success = test_func()
            elapsed_time = time.time() - start_time
            
            self.test_results[test_name] = success
            self.test_times[test_name] = elapsed_time
            
            status = "PASSED" if success else "FAILED"
            logger.info(f"Test {test_name}: {status} (took {elapsed_time:.2f}s)")
            return success
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.test_results[test_name] = False
            self.test_times[test_name] = elapsed_time
            
            logger.error(f"Test {test_name}: FAILED with exception: {e}")
            logger.error(f"Exception details:", exc_info=True)
            return False
    
    def print_summary(self):
        """Print test results summary."""
        logger.info(f"\n{'='*60}")
        logger.info("TEST RESULTS SUMMARY")
        logger.info(f"{'='*60}")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        logger.info("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "PASS" if result else "FAIL"
            time_str = f"{self.test_times[test_name]:.2f}s"
            logger.info(f"  {test_name:<30} {status:<6} {time_str}")
        
        return failed_tests == 0


def test_basic_initialization():
    """Test basic camera initialization and cleanup."""
    logger.info("Testing basic camera initialization...")
    
    # Test simple camera creation
    camera = create_simple_camera(
        exposure_time_ms=100.0,
        output_dir="test_output/basic"
    )
    
    try:
        # Test initialization
        if not camera.initialize():
            logger.error("Camera initialization failed")
            return False
        
        logger.info("✓ Camera initialized successfully")
        
        # Test API version
        version = camera.get_api_version()
        logger.info(f"✓ API Version: {version}")
        
        # Test state tracking
        assert camera.is_initialized == True, "Initialization flag not set"
        assert camera.is_acquiring == False, "Acquisition flag should be False"
        
        logger.info("✓ State tracking working correctly")
        
        return True
        
    finally:
        camera.cleanup()
        logger.info("✓ Camera cleanup completed")


def test_batch_acquisition():
    """Test batch image acquisition mode."""
    logger.info("Testing batch acquisition mode...")
    
    camera = create_simple_camera(
        exposure_time_ms=50.0,
        output_dir="test_output/batch",
        hotpixel_correction=True
    )
    
    try:
        if not camera.initialize():
            return False
        
        # Test batch acquisition
        num_images = 3
        logger.info(f"Acquiring {num_images} images in batch mode...")
        
        success = camera.process_batch_images(num_images)
        
        if success:
            logger.info("✓ Batch acquisition completed successfully")
            
            # Verify files were created
            expected_files = num_images
            fits_dir = os.path.join(camera.output_dir, "fits")
            
            if os.path.exists(fits_dir):
                actual_files = len([f for f in os.listdir(fits_dir) if f.endswith('.fits')])
                logger.info(f"✓ Created {actual_files}/{expected_files} FITS files")
                
                if actual_files >= expected_files:
                    return True
                else:
                    logger.error(f"Expected {expected_files} files, got {actual_files}")
                    return False
            else:
                logger.error("FITS output directory not found")
                return False
        else:
            logger.error("Batch acquisition failed")
            return False
            
    finally:
        camera.cleanup()


def test_continuous_acquisition():
    """Test continuous acquisition mode."""
    logger.info("Testing continuous acquisition mode...")
    
    camera = create_advanced_camera(
        exposure_time_ms=20.0,
        output_dir="test_output/continuous",
        enable_jpeg_output=True
    )
    
    try:
        if not camera.initialize():
            return False
        
        if not camera.start_acquisition():
            logger.error("Failed to start acquisition")
            return False
        
        logger.info("✓ Acquisition started successfully")
        
        # Test GPS coordinate update
        camera.update_gps_coordinates(25.0330, 121.5654)
        logger.info("✓ GPS coordinates updated")
        
        # Test continuous image acquisition
        num_images = 5
        successful_captures = 0
        
        for i in range(num_images):
            logger.info(f"Acquiring image {i+1}/{num_images}...")
            
            image_data, buffer = camera.acquire_single_image(timeout_ms=5000)
            
            if image_data is not None:
                successful_captures += 1
                logger.info(f"✓ Image {i+1} acquired: shape {image_data.shape}")
                
                # Test metadata update
                camera.update_analysis_results(
                    measured_alt=45.0 + i,
                    measured_azi=180.0 + i*2,
                    head_diff=i*0.1
                )
                
                # Test saving
                if camera.save_fits(image_data, i+1):
                    logger.info(f"✓ FITS file saved for image {i+1}")
                
                if camera.save_jpeg(image_data, i+1):
                    logger.info(f"✓ JPEG file saved for image {i+1}")
                
                # Queue buffer back
                if buffer:
                    camera.queue_buffer(buffer)
                    
            else:
                logger.warning(f"Failed to acquire image {i+1}")
            
            time.sleep(0.1)
        
        logger.info(f"✓ Successfully captured {successful_captures}/{num_images} images")
        
        return successful_captures >= (num_images * 0.8)  # Allow 20% failure rate
        
    finally:
        camera.cleanup()


def test_context_manager():
    """Test context manager functionality."""
    logger.info("Testing context manager...")
    
    try:
        with CameraAcquisition(
            exposure_time_ms=100.0,
            output_dir="test_output/context",
            enable_jpeg_output=False
        ) as camera:
            
            logger.info("✓ Camera initialized via context manager")
            
            # Test that camera is initialized
            assert camera.is_initialized == True, "Camera should be initialized"
            
            # Test batch processing
            success = camera.process_batch_images(2)
            
            if success:
                logger.info("✓ Batch processing successful in context manager")
                return True
            else:
                logger.error("Batch processing failed in context manager")
                return False
                
    except Exception as e:
        logger.error(f"Context manager test failed: {e}")
        return False


def test_error_handling():
    """Test error handling and recovery."""
    logger.info("Testing error handling...")
    
    camera = CameraAcquisition(
        exposure_time_ms=50.0,
        output_dir="test_output/error_handling"
    )
    
    try:
        # Test operations without initialization
        logger.info("Testing operations without initialization...")
        
        # These should fail gracefully
        result = camera.start_acquisition()
        assert result == False, "start_acquisition should fail without initialization"
        logger.info("✓ start_acquisition failed gracefully without initialization")
        
        image_data, buffer = camera.acquire_single_image()
        assert image_data is None, "acquire_single_image should return None without acquisition"
        logger.info("✓ acquire_single_image failed gracefully without acquisition")
        
        # Test double initialization
        logger.info("Testing double initialization...")
        camera.initialize()
        result = camera.initialize()  # Should return True without error
        assert result == True, "Double initialization should succeed"
        logger.info("✓ Double initialization handled correctly")
        
        # Test double acquisition start
        logger.info("Testing double acquisition start...")
        camera.start_acquisition()
        result = camera.start_acquisition()  # Should return True without error
        assert result == True, "Double acquisition start should succeed"
        logger.info("✓ Double acquisition start handled correctly")
        
        return True
        
    finally:
        camera.cleanup()


def test_stress_acquisition():
    """Stress test with rapid acquisition."""
    logger.info("Running stress test with rapid acquisition...")
    
    camera = create_advanced_camera(
        exposure_time_ms=10.0,  # Very short exposure
        output_dir="test_output/stress"
    )
    
    try:
        if not camera.initialize():
            return False
        
        if not camera.start_acquisition():
            return False
        
        # Rapid acquisition test
        num_images = 50
        successful_captures = 0
        start_time = time.time()
        
        for i in range(num_images):
            image_data, buffer = camera.acquire_single_image(timeout_ms=1000)
            
            if image_data is not None:
                successful_captures += 1
                # Queue buffer immediately
                if buffer:
                    camera.queue_buffer(buffer)
            
            # No delay - acquire as fast as possible
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        fps = successful_captures / elapsed_time
        success_rate = (successful_captures / num_images) * 100
        
        logger.info(f"✓ Stress test completed:")
        logger.info(f"  - Images captured: {successful_captures}/{num_images}")
        logger.info(f"  - Success rate: {success_rate:.1f}%")
        logger.info(f"  - Time elapsed: {elapsed_time:.2f}s")
        logger.info(f"  - Average FPS: {fps:.2f}")
        
        # Consider test successful if we get at least 70% success rate
        return success_rate >= 70.0
        
    finally:
        camera.cleanup()


def test_performance_benchmark():
    """Performance benchmark test."""
    logger.info("Running performance benchmark...")
    
    camera = create_advanced_camera(
        exposure_time_ms=20.0,
        output_dir="test_output/benchmark"
    )
    
    try:
        # Benchmark initialization
        start_time = time.time()
        if not camera.initialize():
            return False
        init_time = time.time() - start_time
        
        # Benchmark acquisition start
        start_time = time.time()
        if not camera.start_acquisition():
            return False
        acq_start_time = time.time() - start_time
        
        # Benchmark image acquisition
        num_samples = 10
        acquisition_times = []
        
        for i in range(num_samples):
            start_time = time.time()
            image_data, buffer = camera.acquire_single_image(timeout_ms=3000)
            acq_time = time.time() - start_time
            
            if image_data is not None:
                acquisition_times.append(acq_time)
                if buffer:
                    camera.queue_buffer(buffer)
        
        if acquisition_times:
            avg_acq_time = np.mean(acquisition_times)
            min_acq_time = np.min(acquisition_times)
            max_acq_time = np.max(acquisition_times)
            
            logger.info(f"✓ Performance benchmark results:")
            logger.info(f"  - Initialization time: {init_time:.3f}s")
            logger.info(f"  - Acquisition start time: {acq_start_time:.3f}s")
            logger.info(f"  - Average acquisition time: {avg_acq_time:.3f}s")
            logger.info(f"  - Min acquisition time: {min_acq_time:.3f}s")
            logger.info(f"  - Max acquisition time: {max_acq_time:.3f}s")
            logger.info(f"  - Theoretical max FPS: {1/avg_acq_time:.1f}")
            
            return True
        else:
            logger.error("No successful acquisitions for benchmark")
            return False
            
    finally:
        camera.cleanup()


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="IDS Camera Testing Script")
    parser.add_argument("--test", 
                       choices=["basic", "batch", "continuous", "context", "error", "stress", "benchmark", "all"],
                       default="basic",
                       help="Type of test to run")
    parser.add_argument("--output", type=str, default="test_output",
                       help="Output directory for test files")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create test suite
    test_suite = CameraTestSuite(args.output)
    
    # Define available tests
    tests = {
        "basic": test_basic_initialization,
        "batch": test_batch_acquisition,
        "continuous": test_continuous_acquisition,
        "context": test_context_manager,
        "error": test_error_handling,
        "stress": test_stress_acquisition,
        "benchmark": test_performance_benchmark
    }
    
    logger.info(f"Starting IDS Camera Test Suite")
    logger.info(f"Output directory: {args.output}")
    
    # Run selected tests
    if args.test == "all":
        # Run all tests
        for test_name, test_func in tests.items():
            test_suite.run_test(test_name, test_func)
    else:
        # Run specific test
        if args.test in tests:
            test_suite.run_test(args.test, tests[args.test])
        else:
            logger.error(f"Unknown test: {args.test}")
            return 1
    
    # Print summary
    all_passed = test_suite.print_summary()
    
    if all_passed:
        logger.info("\n🎉 All tests passed!")
        return 0
    else:
        logger.error("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
