#!/usr/bin/env python3
"""
Test runner for Lillith test suite
Runs all unit and integration tests and generates a report.
"""

import sys
sys.dont_write_bytecode = True

import unittest
import time
from io import StringIO


def run_all_tests():
    """Run all tests and generate report"""
    print("="*70)
    print("LILLITH TEST SUITE")
    print("="*70)
    print()
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = '.'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print()
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Time: {end_time - start_time:.2f} seconds")
    print("="*70)
    
    # Return exit code
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
