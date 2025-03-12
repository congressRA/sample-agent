#!/bin/bash

# Generate a short random UUID suffix (8 characters)
HASH=$(openssl rand -hex 4)
echo "Using hash: $HASH for this batch"

# Loop through congress sessions 113-118
for congress in {113..118}
do
  echo "Processing Congress $congress..."
  
  # Define output files with hash suffix
  CLUSTERS_FILE="clusters_congress_${congress}_${HASH}.json"
  GRIDLOCK_FILE="gridlock_score_congress_${congress}_${HASH}.json"
  
  # Run the first command to generate clusters
  echo "Running: python nyt_clusters.py --congress $congress --output $CLUSTERS_FILE"
  python nyt_clusters.py --congress $congress --output $CLUSTERS_FILE
  
  # Check if the previous command succeeded
  if [ $? -ne 0 ]; then
    echo "Error generating clusters for Congress $congress. Continuing to next congress."
    continue
  fi
  
  # Run the second command to analyze the bills
  echo "Running: python bill_analyzer.py --clusters-file $CLUSTERS_FILE --congress $congress --output $GRIDLOCK_FILE"
  python bill_analyzer.py --clusters-file $CLUSTERS_FILE --congress $congress --output $GRIDLOCK_FILE
  
  # Check if the previous command succeeded
  if [ $? -ne 0 ]; then
    echo "Error analyzing bills for Congress $congress."
    continue
  fi
  
  echo "Completed processing for Congress $congress"
  echo "------------------------"
done

echo "All analyses complete. Output files have suffix: $HASH"
echo "Cluster files: clusters_congress_XXX_${HASH}.json"
echo "Gridlock score files: gridlock_score_congress_XXX_${HASH}.json"
