"""
Created on Wed Sept 10 8:26:30 2025
Author: Koen Gerrits
Do not spread this script without the authors approval.
"""

# import packages
import logging
import warnings
import pandas as pd
import joblib
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry

logging.basicConfig(filename='Collecting_weather_forecast.log',
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
print("Initiating module")

def get_model_data(lat: float, lon: float, model_name: str, forecast_days: int) -> pd.DataFrame:
    """Retrieve hourly forecast + past data from Open-Meteo API."""

    URL = "https://api.open-meteo.com/v1/forecast"

    COLUMNS = ["relative_humidity_2m", "temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m"]

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(COLUMNS),  # must be a comma-separated string
        "models": model_name,
        "forecast_days": forecast_days,
        "past_days": 14,
    }

    try:
        logging.info("Requesting weather data from Open-Meteo API...")
        responses = openmeteo.weather_api(URL, params=params)
        response = responses[0]

        hourly = response.Hourly()
        if hourly is None:
            logging.warning("No hourly data returned from API")
            return pd.DataFrame()

        # Build dataframe
        data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )
        }

        for idx, col in enumerate(COLUMNS):
            try:
                data[col] = hourly.Variables(idx).ValuesAsNumpy()
            except Exception as e:
                logging.warning(f"Missing data for column {col}: {e}")
                data[col] = None

        df = pd.DataFrame(data)
        logging.info("Successfully retrieved and parsed weather data")
        return df

    except Exception as e:
        logging.exception(f"Exception occurred while fetching weather data: {e}")
        return pd.DataFrame()

def calculate_rolling_sums( weather_data: pd.DataFrame, weather_vars: list, windows: list ):
    """
    Calculates multiple rolling window summations for weather variables.
    """
    # Create rolling sum columns
    for var in weather_vars:
        for w in windows:
            weather_data[f'{var}_som_{w}'] = weather_data[var].rolling(window=w, min_periods=1).sum()
    return weather_data

# Function to calculate Es (Saturated Vapor Pressure)
def calculate_Es(T):
    return 6.112 * np.exp((17.62 * T) / (T + 243.12))

def calculate_hdw(df):
    """
    Calculates the Saturated Vapor Pressure (Es) and Vapor Pressure (E)
    from a DataFrame containing columns for Mean Temperature and Relative Humidity.

    Args:
    df (pd.DataFrame): DataFrame with 'Mean Temperature' (°C) and 'Relative Humidity' (%).

    Returns:
    pd.DataFrame: DataFrame with added columns for 'Saturated Vapor Pressure (Es)'
                    and 'Vapor Pressure (E)'.
    """
    # Apply the function to the 'Mean Temperature' column to get Es
    df['Saturated Vapor Pressure (Es)'] = df['temp_gem'].apply(calculate_Es)

    # Calculate the Vapor Pressure (E) using the Relative Humidity
    df['Vapor Pressure (E)'] = (df['vocht_gem'] / 100) * df['Saturated Vapor Pressure (Es)']

    df["VPD"] = df['Saturated Vapor Pressure (Es)'] - df["Vapor Pressure (E)"]
    df["HDWI"] = df["VPD"] * df["wind_gem"]
    df["HDWI_gust"] = df["VPD"] * df["wind_max"]
    return df

# Filter is_sparse warning
warnings.simplefilter(action='ignore', category=FutureWarning)
# Initiate class with logging
try:
    logging.info("Initiate logging for weather data collection")
except (ValueError, TypeError) as e:
    print(f"An error occurred: {e}")
    logging.error("Error initiating weather data collection: %s", e)
except Exception as e:
    logging.error("Error initiating weather data collection: %s", e)

# Set up cached session with retries
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Load meteoserver data
try:
    print("Retrieving weather data")
    # Get weather data for Deelen
    data = get_model_data(lat=52.06044892238738, lon=5.8875201130685255, model_name="knmi_seamless", forecast_days=7)
except requests.HTTPError as e:
    logging.error("Failed to load GFS data. Status code: %s", e.response.status_code)
except requests.RequestException as e:
    logging.error("An error occurred during the request to load data: %s", e)
except Exception as e:
    logging.error("An unexpected error occurred while loading weather data: %s", e)


try:
    print("Transforming data")
    data["Datum"] = data["date"].dt.date

    daily_df = data.groupby(["Datum"]).agg(
        temp_gem=("temperature_2m", "mean"),
        temp_min=("temperature_2m", "min"),
        temp_max=("temperature_2m", "max"),
        vocht_gem=("relative_humidity_2m", "mean"),
        vocht_max=("relative_humidity_2m", "max"),
        vocht_min=("relative_humidity_2m", "min"),
        neerslag_som=("precipitation", "sum"),
        wind_gem=("wind_speed_10m", "mean"),
        wind_max=("wind_gusts_10m", "max"),
        wind_min=("wind_speed_10m", "min")
    ).reset_index()

    # Select relevant columns  
    final = daily_df[["Datum", "temp_gem", "temp_min", "temp_max", "vocht_gem", "vocht_max",
                      "vocht_min", "neerslag_som", "wind_gem", "wind_max", "wind_min"]]
    logging.info("Successfully transformed the data")
except ValueError as e:
    print(f"ValueError occured: {e}")
    logging.error("ValueError occured while transforming the data: %s", e)
except TypeError as e:
    print(f"TypeError occured: {e}")
    logging.error("TypeError occured while transforming the data: %s", e)
except Exception as e:
    print(f"An error occured: {e}")
    logging.error("An error occured while transforming the data: %s", e)

# List of columns to calculate rolling sums for
weather_vars = ['temp_gem', 'vocht_gem', 'neerslag_som']

# Rolling windows in days
windows = [3, 7]

final = calculate_rolling_sums(weather_data=final, weather_vars=weather_vars, windows=windows)

# Rename odd columns
final.rename(columns={"neerslag_som_som_3": "neerslag_som_3", "neerslag_som_som_7" : "neerslag_som_7"}, inplace=True)

final = calculate_hdw(final)

# try:
#     # Formulate correct feature order
#     features = [
#     "wind_max",
#     "wind_gem",
#     "temp_gem",
#     "temp_gem_som_3",
#     "temp_gem_som_7",
#     "temp_min",
#     "temp_max",
#     "vocht_gem",
#     "vocht_max",
#     "neerslag_som_3",
#     "neerslag_som_7",
#     "vocht_min",
#     "neerslag_som",
#     "HDWI",
#     "HDWI_gust",
#     "vocht_gem_som_3",
#     "vocht_gem_som_7"]

#     # load XGB model
#     MODEL_PATH = "<insert_your_model_path_here>"
#     loaded_model = joblib.load(MODEL_PATH)
#     SCALER_PATH = "<insert_your_scaler_path_here>"
#     scaler = joblib.load(SCALER_PATH)
#     transformed_data = scaler.transform(final[features])
#     # Run values through model
#     final["predictions"] = loaded_model.predict_proba(transformed_data)[:,1]
#     logging.info("Successfully calculated predictions")
# except ValueError as e:
#     print(f"ValueError occured: {e}")
#     logging.error("ValueError occured while calculating predictions: %s", e)
# except TypeError as e:
#     print(f"TypeError occured: {e}")
#     logging.error("TypeError occured while calculating predictions: %s", e)
# except Exception as e:
#     print(f"An error occured: {e}")
#     logging.error("An error occured while calculating predictions: %s", e)


