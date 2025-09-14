# Project "Optimal Routes": Trip Duration and Traffic Congestion Prediction

A mini web service that predicts trip duration and visualizes traffic congestion for two alternative route options.

## About The Project

This project is an end-to-end Data Science solution, from data collection to the deployment of an interactive web application. The main objective is to provide the user with two optimal paths for a selected destination, complete with a time forecast and a visual map of road congestion. The service is powered by a cascade of two machine learning models.

## Demo

An interactive version of the application is deployed on Hugging Face Spaces and is available at the following link:

**[https://huggingface.co/spaces/ErzhanAb/Optimal-routes]**

## Tech Stack

- **Machine Learning:** Scikit-learn, XGBoost, CatBoost
- **Data Handling:** Pandas, NumPy
- **Web Interface & Visualization:** Gradio, Folium
- **Geospatial Data:** Shapely

## Data Collection and Preparation

The data for the project was collected using the **2GIS API** over a period of 12 days: **from September 3rd (Wednesday) to September 14th (Sunday)**. Data was gathered approximately every hour for 10 popular routes. This resulted in two main datasets:
- `segments.csv`: Contained information about traffic congestion ("coloring") on individual short road segments. Used for training Model 1.
- `routes.csv`: Contained general route information (duration, distance, geometry). Used for training Model 2.

To standardize the inference stage, a reference file `canonical_routes.json` was created. For each of the two route variants, a **medoid** (the most typical, real-world track) was identified using the **FastDTW** algorithm, which was then divided into 50-meter segments.

## Machine Learning Models

### Model 1: Segment Congestion Prediction (XGBoost)
- **Task:** Multiclass classification (4 classes: `fast`, `normal`, `slow`, `slow-jams`).
- **Input Features:** Segment coordinates, cyclical features for time of day and day of the week, one-hot encoded route name.
- **Result:** The model achieved an **F1-score (macro) of 0.71** and a **ROC AUC of 0.93** on the test set.

### Model 2: Trip Duration Prediction (CatBoost)
- **Task:** Regression (predicting time in seconds).
- **Input Features:** Total distance, number of maneuvers, cyclical time features, and **aggregated predictions from Model 1** (the share of segments with different congestion levels).
- **Result:** The model demonstrated high accuracy with an **R² of 0.967** and a Mean Absolute Error (MAE) of approximately **66 seconds** (~1.1 minutes).

## Local Deployment

Follow these steps to run the application on your local machine.

**Important:** This project uses Git LFS to store large model files. Make sure you have Git LFS installed.

1.  **Clone the repository:**
    ```bash
    git lfs install
    git clone https://github.com/ErzhanAb/Optimal-Routes-10.git
    cd Optimal-Routes-10
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python app.py
    ```
    After launching, a link (usually `http://127.0.0.1:7860`) will appear in the terminal. Open it in your web browser.

## Repository Structure

| File / Directory          | Description                                           |
| ------------------------- | ----------------------------------------------------- |
| `app.py`                  | Main Gradio application script                        |
| `canonical_routes.json`   | Reference file with standard segmented routes         |
| `catboost_eta_predictor.cbm`| CatBoost model for duration prediction                |
| `xgboost_traffic_model...`| XGBoost model for traffic prediction                  |
| `requirements.txt`        | List of dependencies for pip                          |
