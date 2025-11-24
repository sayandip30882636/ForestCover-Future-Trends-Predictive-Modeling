# Forest Cover Prediction in India

## Project Overview

This project aims to analyze historical forest cover data for various states in India and predict future trends using different time series and machine learning models. The goal is to provide insights into forest area changes and to compare the performance of various predictive techniques.

## Data Sources

The analysis utilizes two primary datasets:
1.  **Forest Coverage (1987-2013).xlsx**: Contains state-wise forest cover data from 1987 to 2013, with biennial reports.
2.  **Forest.xlsx**: Provides additional forest-related attributes for Indian states.
3.  **India GIS data**: Shapefiles (`India_boundary.shp` and `Indian_states.shp`) for geographical representation of India and its states.

## Methodology

### 1. Data Loading and Preprocessing
-   Historical forest data from Excel files are loaded and merged.
-   State names are standardized for consistency across datasets.
-   The wide-format data is converted to a long format suitable for time series analysis.
-   Missing values (represented by '-') are handled using interpolation.
-   The data is merged with geographical information system (GIS) data to incorporate state boundaries and calculate actual geographical areas.
-   Feature engineering is performed to create lagged features, forest growth rates, and rolling means of forest area.

### 2. Exploratory Data Analysis (EDA)
-   Correlation heatmaps are generated to understand relationships between features.
-   Time series plots visualize forest area trends for selected states.
-   Distribution of forest growth rates is examined.

### 3. Predictive Modeling

#### A. ARIMA (AutoRegressive Integrated Moving Average) Model
-   **Approach**: An ARIMA(1,1,1) model is applied to the time series of forest area for each state independently.
-   **Forecasting**: The model forecasts forest area for 10 future biennial periods (up to 2033).
-   **Visualization**: Forecasts are plotted against historical actual data for selected states.

#### B. Random Forest Regressor
-   **Approach**: A Random Forest Regressor is trained to predict forest area based on engineered features (lagged forest area, rolling mean, growth rate, and forest percentage).
-   **Training**: The data is split into training and testing sets, and the model is trained on historical data.
-   **Evaluation**: Model performance is assessed using R² score and Mean Absolute Error (MAE).
-   **Prediction**: The trained model makes predictions for the latest available year (2013 data was used as input to predict the _next_ year's value, effectively for 2015 based on the way features are lagged).

#### C. Linear Mixed Effects Model (LMM)
-   **Approach**: A Linear Mixed Effects Model is used to analyze the overall trend in forest area (`forest_area ~ year`) while accounting for state-specific variability (random intercepts grouped by state).
-   **Interpretation**: The model provides insights into the average annual change in forest area across all states (fixed effect) and the extent of state-to-state heterogeneity (random effect variance).

### 4. Geospatial Visualization and Comparison

-   **Predicted Forest Cover Maps**: Geospatial plots are created using `geopandas` to visualize the predicted forest cover percentage for each state based on:
    -   Random Forest predictions (using the latest available historical data).
    -   ARIMA forecasts for a specific future year (e.g., 2015).
-   **Model Comparison**: A side-by-side comparison of forest cover percentage predictions from Random Forest and ARIMA models is presented.
-   **Difference Map**: A map illustrating the difference in predicted forest cover percentage between the two models is generated to highlight regions where predictions diverge significantly.

## Results Highlights

-   The Random Forest model achieved a high R² score of 0.998 and a low MAE of 422.7, indicating strong performance in predicting forest area based on historical features.
-   ARIMA forecasts provide state-specific future trends in forest cover.
-   The LMM analysis suggests a marginal average annual decrease in forest area across states, but with significant state-level variations, indicating that a one-size-fits-all trend does not capture the full picture.
-   Geospatial visualizations help in understanding the spatial distribution of forest cover predictions and the discrepancies between different modeling approaches.

## How to Run the Project

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Install Dependencies**:
    The project relies on libraries such as `pandas`, `numpy`, `geopandas`, `rasterio`, `rioxarray`, `rasterstats`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`, `pmdarima`, `xgboost`, `lightgbm`, `optuna`, `shap`, and `prophet`.
    All required packages can be installed using the `pip` command provided in the notebook:
    ```bash
    !pip install rioxarray rasterio rasterstats folium matplotlib scikit-learn \
    xgboost lightgbm statsmodels pmdarima optuna shap prophet
    ```
3.  **Prepare Data**: Ensure the `Forest Data` and `India GIS data` directories are correctly placed relative to the notebook, or update the file paths in `COLAB CELL 2`.
4.  **Run the Notebook**: Open and execute the Jupyter/Colab notebook cells sequentially.

## Future Work

-   Incorporate external factors like climate data, population density, and socio-economic indicators.
-   Explore more advanced time series models or hybrid approaches.
-   Improve handling of missing data and outliers.
-   Implement hyperparameter tuning for all models to optimize performance.
