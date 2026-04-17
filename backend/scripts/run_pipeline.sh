# CareerAlign Pipeline Runner
#
# This script runs the full data pipeline or a specific step.
#
# USAGE:
#
#   Run full pipeline 
#       ./run_pipeline.sh
#
#   Run a specific step: 
#       ./run_pipeline.sh --step "Embed Jobs" (see all steps in app/scripts/run_pipeline.py)
#
#   Show help:
#       ./run_pipeline.sh --help
#
# NOTES:
# - Make sure your virtual environment is activated before running
# - Ensure all environmental variables are set (see README for details)
# - Script will stop if any step fails

# Exit immediately if a command exits with a non-zero status set -e
set -e

# Help function to display usage
show_help() {
    echo ""
    echo "CareerAlign Pipeline Runner"
    echo ""
    echo "Usage:" 
    echo " ./run_pipeline.sh Run full pipeline" 
    echo " ./run_pipeline.sh --step \"NAME\" Run a specific step" 
    echo " ./run_pipeline.sh --help Show this help message" 
    echo "" 
    echo "Example:" 
    echo " ./run_pipeline.sh --step \"Embed Jobs\"" 
    echo "" 
}

# Parse arguments
STEP="" 

while [[ "$#" -gt 0 ]]; do 
    case $1 in 
        --step) 
            STEP="$2" 
            shift 2 
            ;; 
        --help) 
            show_help 
            exit 0 
            ;; 
        *) 
            echo "Unknown argument: $1" 
            show_help 
            exit 1 
            ;; 
        esac 
    done

# Run pipeline
echo "Starting pipeline..."

if [ -z "$STEP" ]; then
    echo "Running full pipeline..."
    python -m backend.pipelines.run_pipeline
else
    echo "Running step: $STEP"
    python -m backend.pipelines.run_pipeline --step "$STEP"
fi

echo "Pipeline execution complete."