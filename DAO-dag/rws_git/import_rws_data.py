"""
Data retrieval RWS

Created on Thu Jul 25 08:15:36 2024

@author: Koen Gerrits
@email: k.gerrits@vnog.nl

Do not share this script without the author's approval.'
"""
import logging
import requests
import json
import pandas as pd
from datetime import datetime, timedelta


# %% Initiate path variables
# Initiating logging

logging.basicConfig(filename='collect_rws_data.log',
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Define the location of the required json files here
path = "C:/Users/gerritsk/Documents/RWS_data/Waterhoogte/Oost5_locaties/"

locations = "locations_oost_5.json"
classification = "classifications_oost_5.json"

# Define the URL for the API endpoint
url = "https://waterwebservices.rijkswaterstaat.nl/ONLINEWAARNEMINGENSERVICES_DBO/OphalenWaarnemingen"

# %% Formulate definitions

# Create def to make proper datetime string required for API
def date_string_last_week():
    today = datetime.now()
    now = f"{today.year}-{today.month}-{today.day}T{today.hour}:{today.minute}:{today.second}.000+01:00"
    week_ago = today - timedelta(weeks=1)
    last_week = f"{week_ago.year}-{week_ago.month}-{week_ago.day}T00:00:00.000+01:00"
    return last_week, now

def date_string_next_week():
    today = datetime.now()
    now = f"{today.year}-{today.month}-{today.day}T{today.hour}:{today.minute}:{today.second}.000+01:00"
    two_days = today + timedelta(days=2)
    next_week = f"{two_days.year}-{two_days.month}-{two_days.day}T{two_days.hour}:{two_days.minute}:{two_days.second}.000+01:00"
    return now, next_week

# Function to classify water level
def classify_water_level(location, water_level, classification_data):
    for classification, lower_bound, upper_bound in classification_data[location]:
        if lower_bound <= water_level < upper_bound:
            return classification
    return None

def retrieve_rws_data(body, location, url):
    # Make the POST request
    response = requests.post(url, json=body)
    # Check if the request was successful
    if response.status_code == 200:
        print(f"Request for {location} was successful!")
    else:
        print(f"Request for {location} failed with status code:", response.status_code)
        print("Response message:", response.text)
    
    response_dict = json.loads(response.content.decode('utf-8'))
    if response_dict["Succesvol"] == False:
        print(f"Something went wrong: {response_dict['Foutmelding']}")
    else:
        data = response_dict.get('WaarnemingenLijst', [])
        records = []
        for item in data:
            for measurement in item.get('MetingenLijst', []):
                tijdstip = measurement.get('Tijdstip')
                waarde_numeriek = measurement.get('Meetwaarde', {}).get('Waarde_Numeriek')
                records.append({'Tijdstip': tijdstip, 'Waarde_Numeriek': waarde_numeriek})
        # Create a DataFrame
        df = pd.DataFrame(records)
        df["Tijdstip"] = pd.to_datetime(df["Tijdstip"])
        df["Locatie"] = location
        return df
    
# %% data preparation

# Load json files
with open(path+classification) as json_file:
        classification_data = json.load(json_file)

with open(path+locations) as json_file:
    location_data = json.load(json_file)
    
# %% Retrieve data from all locations
    
# Initiate logging
# try:
#     logging.info("Successfully initiated rws data collection")
# except (ValueError, TypeError) as e:
#     print(f"An error occurred: {e}")
#     logging.error("Error initiating rws data collection: %s", e)
# except Exception as e:
#     logging.error("Error initiating rws data collection: %s", e)

code = "WATHTE"
afgelopen_week = pd.DataFrame()

try:
    print("Ophalen historie waterhoogtes...")
    for location in location_data:
        body = {"Locatie":location_data[location],
             "AquoPlusWaarnemingMetadata":{
               "AquoMetadata":{"Compartiment":{"Code":"OW"}, "Grootheid":{"Code":code}}},
             "Periode":{
               "Begindatumtijd":date_string_last_week()[0],
               "Einddatumtijd":date_string_last_week()[1]}}
        df = retrieve_rws_data(body, location, url)

        afgelopen_week = pd.concat([afgelopen_week, df])
    logging.info("Ophalen historie succesvol.")
except requests.HTTPError as e:
    logging.error("Failed to load history data. Status code: %s", e.response.status_code)
except requests.RequestException as e:
    logging.error("An error occurred during the request to load history data: %s", e)
except Exception as e:
    logging.error("An unexpected error occurred while loading history data: %s", e)

# %% Waterstanden verwachtingen

code = "WATHTEVERWACHT"
verwachting = pd.DataFrame()
try:
    print("Ophalen verwachting waterhoogtes...")
    for location in location_data:
        body = {"Locatie":location_data[location],
             "AquoPlusWaarnemingMetadata":{
               "AquoMetadata":{"Compartiment":{"Code":"OW"}, "Grootheid":{"Code":code}}},
             "Periode":{
               "Begindatumtijd":date_string_next_week()[0],
               "Einddatumtijd":date_string_next_week()[1]}}
        df = retrieve_rws_data(body, location, url)
        verwachting = pd.concat([verwachting, df])
    logging.info("Ophalen verwachtingen succesvol.")
except requests.HTTPError as e:
    logging.error("Failed to load history data. Status code: %s", e.response.status_code)
except requests.RequestException as e:
    logging.error("An error occurred during the request to load history data: %s", e)
except Exception as e:
    logging.error("An unexpected error occurred while loading history data: %s", e)

# %% Samenvoegen verwachting en historie
try:
    # Samenvoegen historie en verwachting
    waterhoogte = pd.concat([afgelopen_week, verwachting]).sort_values(by="Tijdstip")
    # Waarden classificeren
    waterhoogte["Classification"] = waterhoogte.apply(lambda row: classify_water_level(row["Locatie"], row["Waarde_Numeriek"], classification_data), axis=1)
    
    # Filter missing values (999999)
    waterhoogte = waterhoogte[waterhoogte["Waarde_Numeriek"] < 99999]
    waterhoogte = waterhoogte[waterhoogte["Waarde_Numeriek"] != 999.99]
    # Filter on whole hours
    waterhoogte = waterhoogte[waterhoogte["Tijdstip"].dt.minute == 0]
except ValueError as e:
    logging.error("A value error occurred while combining data: %s", e)
except Exception as e:
    logging.error("An error occurred while combining data: %s", e)



# %% Debiet afgelopen week

with open(path+'debiet_locations.json') as json_file:
    Debiet_locations = json.load(json_file)

with open(path+'debiet_classification.json') as json_file:
    Debiet_classification = json.load(json_file)

try:
    print("Ophalen Debiet historie...")
    q_afgelopen_week = pd.DataFrame()
    for location in Debiet_locations:
        body = {"Locatie": Debiet_locations[location],
             "AquoPlusWaarnemingMetadata":{
               "AquoMetadata":{"Compartiment":{"Code":"OW"}, "Grootheid":{"Code":"Q"}}},
             "Periode":{
               "Begindatumtijd":date_string_last_week()[0],
               "Einddatumtijd":date_string_last_week()[1]}}
        q_result = retrieve_rws_data(body, location, url)
        q_afgelopen_week = pd.concat([q_afgelopen_week, q_result])
    logging.info("Ophalen verwachtingen succesvol.")
except requests.HTTPError as e:
    logging.error("Failed to load history data. Status code: %s", e.response.status_code)
except requests.RequestException as e:
    logging.error("An error occurred during the request to load history data: %s", e)
except Exception as e:
    logging.error("An unexpected error occurred while loading history data: %s", e)

# %% Debiet verwachting
try:
    print("Ophalen debiet verwachting...")
    q_verwachting = pd.DataFrame()
    for location in Debiet_locations:
        body = {"Locatie": Debiet_locations[location],
             "AquoPlusWaarnemingMetadata":{
               "AquoMetadata":{"Compartiment":{"Code":"OW"}, "Grootheid":{"Code":"QVERWACHT"}}},
             "Periode":{
               "Begindatumtijd":date_string_next_week()[0],
               "Einddatumtijd":date_string_next_week()[1]}}
        q_verwacht_result = retrieve_rws_data(body, location, url)
        q_verwachting = pd.concat([q_verwachting, q_verwacht_result])
    logging.info("Ophalen verwachtingen succesvol.")
except requests.HTTPError as e:
    logging.error("Failed to load history data. Status code: %s", e.response.status_code)
except requests.RequestException as e:
    logging.error("An error occurred during the request to load history data: %s", e)
except Exception as e:
    logging.error("An unexpected error occurred while loading history data: %s", e)

# %% Samenvoegen verwachting en historie debiet
try:
    # Samenvoegen historie en verwachting
    waterafvoer = pd.concat([q_afgelopen_week, q_verwachting]).sort_values(by="Tijdstip")
    # Waarden classificeren
    waterafvoer["Classification"] = waterafvoer.apply(lambda row: classify_water_level(row["Locatie"], row["Waarde_Numeriek"], Debiet_classification), axis=1)
    
    # Filter missing values (999999)
    waterafvoer = waterafvoer[waterafvoer["Waarde_Numeriek"] != 999.99]
    waterafvoer = waterafvoer[waterafvoer["Waarde_Numeriek"] < 99999]
    waterafvoer = waterafvoer[waterafvoer["Tijdstip"].dt.minute == 0]
except ValueError as e:
    logging.error("A value error occurred while combining data: %s", e)
except Exception as e:
    logging.error("An error occurred while combining data: %s", e)

"""
Hier kun je de twee dataframes wegschrijven
"""

