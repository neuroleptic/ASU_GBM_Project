#!/bin/bash
# Script to run Sequenza analysis on all patients using Docker

# Default values
CORES=4
DATA_DIR=$(pwd)/data
OUTPUT_DIR=$(pwd)/results
REFERENCE_GENOME=/input/reference/genome.fa.gz

# Function to display usage
function usage {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -c, --cores NUM         Number of CPU cores to use (default: 4)"
    echo "  -d, --data-dir PATH     Path to data directory (default: ./data)"
    echo "  -o, --output-dir PATH   Path to output directory (default: ./results)"
    echo "  -r, --reference PATH    Path to reference genome in container (default: /input/reference/genome.fa.gz)"
    echo "  -h, --help              Display this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -c|--cores)
            CORES="$2"
            shift 2
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -r|--reference)
            REFERENCE_GENOME="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create a temporary config file
CONFIG_FILE=$(mktemp)
echo "reference_genome: \"$REFERENCE_GENOME\"" > "$CONFIG_FILE"

# Copy the Snakefile to the current directory if it's not already there
if [ ! -f "Snakefile" ]; then
    cp ./simplified-sequenza-snakefile.py Snakefile
fi

# Run the Docker container
echo "Running Sequenza pipeline with Docker..."
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Reference genome: $REFERENCE_GENOME"
echo "Using $CORES cores"

docker run \
    -v "$DATA_DIR:/input" \
    -v "$OUTPUT_DIR:/output" \
    -v "$(pwd)/Snakefile:/workspace/Snakefile" \
    -v "$CONFIG_FILE:/workspace/config.yaml" \
    -w /workspace \
    msfuji/sequenza-pipeline \
    snakemake \
        --cores "$CORES" \
        --configfile /workspace/config.yaml \
        -p

# Clean up
rm -f "$CONFIG_FILE"

echo "Sequenza pipeline completed!"