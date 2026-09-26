#!/bin/bash
# Quick test script for the pipeline

set -e

echo "Installing package in development mode..."
pip install -e .

echo "Testing CLI..."
hindi-dubbing --help

echo "Testing info command..."
hindi-dubbing info

echo "Done!"