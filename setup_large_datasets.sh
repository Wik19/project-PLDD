#!/bin/bash

# Navigate to the data directory
cd data || exit

# Create the new directory structure
mkdir -p Large_Datasets/PLDM
mkdir -p Large_Datasets/PLDU

# Extract PLDM datasets
echo "Extracting PLDM datasets..."
unzip -q PLDM_train.zip -d Large_Datasets/PLDM/
unzip -q PLDM_test.zip -d Large_Datasets/PLDM/
unzip -q PLDM_test_gt.zip -d Large_Datasets/PLDM/

# Extract PLDU datasets
echo "Extracting PLDU datasets..."
unzip -q PLDU_train.zip -d Large_Datasets/PLDU/
unzip -q PLDU_test.zip -d Large_Datasets/PLDU/
unzip -q PLDU_test_gt.zip -d Large_Datasets/PLDU/

# Move the .lst files to their respective folders
echo "Moving .lst files..."
mv PLDM_*.lst Large_Datasets/PLDM/
mv PLDU_*.lst Large_Datasets/PLDU/

echo "Dataset extraction and organization complete!"
