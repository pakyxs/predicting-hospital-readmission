# <h1 align=center> **Predicting Hospital Readmission in Diabetic Patients using Docker and Flask** </h1>

This repository provides a Dockerized analysis pipeline for predicting hospital readmission in diabetic patients.

## Before project

- [User Stories](https://docs.google.com/document/d/1_BqFKL66q6OwGESEL9LdnIFtoJ4ZN1XUMGONl5F3J0w/edit?usp=sharing)
- [MVP](https://docs.google.com/document/d/10j4t7Dm43bZ5p5VO0pc2pdTAf__GpJXO013awTWhf14/edit?usp=sharing)

## Overview

- Dataset
- Analysis and Model Selection
- Usage
- Endpoints
- Files
- Dependencies
- Credits

## Dataset

The raw data set Diabetes 130-US hospitals for years 1999-2008 Data Set can be found [here](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008#). The data set represents 10 years (1999-2008) of clinical care across 130 U.S. hospitals and integrated delivery networks. It includes more than 50 features representing patient and hospital outcomes. Information was extracted from the database for encounters that met the following criteria.

(1) It is a hospital encounter (a hospital admission).

(2) It is a diabetic encounter, that is, one during which any type of diabetes was entered into the system as a diagnosis.

(3) The length of stay was at least 1 day and at most 14 days.

(4) Laboratory tests were performed during the meeting.

(5) Medications were administered during the encounter. The data contains attributes such as patient number, race, sex, age, type of admission, time in hospital, medical specialty of the admitting physician, number of laboratory tests performed, HbA1c test result, diagnosis, number of medications , diabetes medications, number of outpatients. , hospitalization and emergency visits in the year before hospitalization, etc.

### Analysis and Model Selection

The `model.ipynb` notebook details data analysis, cleaning, model training, and comparisons (Linear Regression, XGBoost, SVM). XGBoost achieved the best performance:

- Accuracy: 0.6709
- F1 Score: 0.6695

![XGBoost top 10 parameters](./images/XGBoostfeatures.png)

### Usage

**Prerequisites:** Docker installed.

1. **Clone the Repository**:

   ```bash
   git clone c16-110-n-data-bi
   ```

2. **Building the Docker Image**:

   ```bash
   docker build -t app-readmission .
   ```

3. **Running the Docker Container**:

   ```bash
   docker run -p 5000:5000 app-readmission
   ```

### API Endpoints:

1. **Access Data**:

   - Access Data: http://127.0.0.1:5000/diabetes-data
   - Prediction: http://127.0.0.1:5000/prediction/{input} (Replace {input} with your data)

### Files

- `notebooks`: Contains notebooks with the data analysis and ML model.
- `model.ipynb`: Data analysis and model training notebook.
- `templates`: Contains HTML template for the endpoints.
- `Dockerfile`: Contains instructions for building the Docker image.
- `requirements.txt`: Lists the dependencies required for the analysis.
- `app.py`: Flask application code for the API..
- `README.md`: Instructions for running the analysis pipeline.

### Dependencies

- Flask
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pyarrow
- Optuna
- Xgboost

### Credits

This analysis pipeline was created by: 
- Luciana Agustina Bolo ([LinkedIn](https://www.linkedin.com/in/agustina-bolo/))
- Moreira Rodrigo ([LinkedIn](https://www.linkedin.com/in/rcmoreg/))
- Ezequiel Mazzini([LinkedIn](https://www.linkedin.com/in/ezequiel-mazzini/)).
