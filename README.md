# Neural-Data-Analysis-Python
end-to-end data processing and pattern classification of EEG signals using Python and Random Forest for prosthetic control.

# EEG Signal Processing & Machine Learning for Prosthetic Control
# Overview
This project focuses on the processing and analysis of complex neural signals (EEG) to classify specific patterns for prosthetic control. The goal was to build a robust data pipeline capable of cleaning noisy raw data and applying Machine Learning algorithms to accurately predict and classify patterns.
This work was originally developed during my academic experience at the University "G. d'Annunzio" of Chieti-Pescara.

# Key Technologies
Language: Python
Data Analysis & Cleaning: Pandas, NumPy, SciPy
Machine Learning: Scikit-learn (Random Forest Classifier)
Data Visualization: Matplotlib, Seaborn

# Project Workflow
Data Ingestion & Cleaning: Imported raw EEG datasets and handled missing values, noise, and artifacts.
Signal Processing: Applied filtering techniques to isolate relevant frequency bands.
Feature Extraction: Extracted meaningful statistical and time-frequency features from the cleaned signals.
Model Training: Implemented a Random Forest algorithm to classify the extracted patterns.
Statistical Validation: Evaluated the model's performance using metrics such as Accuracy, Precision, Recall, and Confusion Matrix.

# Results
The Random Forest model successfully classified the EEG patterns with an overall accuracy of 67% and a precision of 75% for the "concentrate" class. The statistical validation confirmed the robustness of the data processing pipeline and the reliability of the extracted features.
! [Grafico Dei Risultati] (Cap_6.1.png)

Note: Due to data privacy and size constraints, only a sample subset of the dataset and the core processing scripts are included in this repository.

# How to Run the Code
Ensure you have Python installed along with the required libraries.

Run the record3.0.py script or open the Jupyter Notebook to view the step-by-step analysis.

# Contatti / About Me
I am an Industrial Engineer (L-9) with a strong background in data analysis, signal processing, and process optimization. I am passionate about applying data-driven approaches to solve complex technological challenges.

LinkedIn: www.linkedin.com/in/morena-troiano-3748a314a
