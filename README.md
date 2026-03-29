# SOM-NeuroFuzzy-EvoFireModel

## Overview
This project implements a hybrid SOM + Fuzzy Logic system for forest fire risk prediction using real-world weather and fire data. Instead of binary classification, the model outputs a continuous fire risk score.

## What It Does
- Integrates fire and weather data
- Handles class imbalance using undersampling
- Uses SOM for micro-climate zoning
- Applies fuzzy logic for risk estimation
- Converts risk to predictions via thresholding

## Key Results
- Baseline model struggles due to imbalance
- Fuzzy model improves fire detection (recall)
- Full dataset shows good recall but higher false positives

## Key Takeaways
- Risk modeling is better than binary classification for this problem
- Data balancing improves sensitivity
- SOM provides structure but needs further tuning
- Model prioritizes recall (important for fire detection)

## How to Run
pip install -r requirements.txt
python src/data_loader.py
python src/balanced.py
python src/baseline_model.py

Run SOM + Fuzzy pipeline via neuroFuzzy.ipynb

## Project Structure
src/
 ├── data_loader.py
 ├── baseline_model.py
 ├── balanced.py
 ├── som_model.py
 ├── fuzzy_model.py

## Conclusion
A practical soft computing approach for fire risk modeling under imbalance. Strong foundation for future improvements.
