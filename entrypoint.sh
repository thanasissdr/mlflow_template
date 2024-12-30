set -e

# Function to log messages
log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Step 1: Run the model loading script
log "Starting model loading process..."
python -m scripts.import_mlflow_into_bentoml

log "Model loading completed successfully."

# Step 2: Start the BentoML server
log "Starting BentoML server..."
exec bentoml serve --port 3000  # Replace with your actual BentoML serve command