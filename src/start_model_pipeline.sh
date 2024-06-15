#!/bin/sh

echo "Starting model training..."
python3 /app/src/train_pipeline.py

echo "Model training complete. Starting prediction..."
python3 /app/src/predict.py

# Keep the container running
tail -f /dev/null