import json
import os
from copy import deepcopy
from datetime import datetime, timedelta
from time import sleep
import sys

import requests as r

MINUTE = 60
MINUTES_OF_SLEEP = 5
SLEEP_TIME = MINUTES_OF_SLEEP*MINUTE
# this will cap the amount of requests per day to 288:
REQUESTS_PER_DAY = (24*MINUTE)/MINUTES_OF_SLEEP
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather?units=metric"
OPENWEATHER_LAT = "lat=43.646882"
OPENWEATHER_LON = "lon=-79.376952"
OPENWEATHER_API_KEY = "35c947ed29a35e7ca48d4da9d7d3cd5b"
TOMTOM_BASE_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
TOMTOM_GARDINER_POINT = "43.642355,-79.379665"
TOMTOM_FRONT_YONGE_POINT = "43.646882,-79.376952"
ZOOM = "zoom=22"

try:
    TOMTOM_API_KEY = os.environ["TOMTOM_API_KEY"]
    OPENWEATHER_API_KEY = os.environ["OPENWEATHER_API_KEY"]
except:
    print("NO API KEY FOUND. EXITING...")
    sys.exit()

requests_performed_per_day = 0
start_at = datetime.now()
curr_datetime = deepcopy(start_at)
end_at = curr_datetime + timedelta(days=33)
traffic_data_retrieved = []
weather_data_retrieved = []

while curr_datetime <= end_at:
    print(f"{datetime.now().isoformat()} - collecting weather data...")
    weather_response = r.get(
        f"{OPENWEATHER_BASE_URL}&{OPENWEATHER_LAT}&{OPENWEATHER_LON}&appid={OPENWEATHER_API_KEY}")
    print(f"{datetime.now().isoformat()} - collected weather data!")
    weather_content = weather_response.content
    weather_content_json = json.loads(weather_content)
    weather_content_json["collected_at"] = curr_datetime.isoformat()

    print(f"{datetime.now().isoformat()} - collecting Gardiner Expy traffic data...")
    gardiner_expy_response = r.get(
        f"{TOMTOM_BASE_URL}?key={TOMTOM_API_KEY}&point={TOMTOM_GARDINER_POINT}&{ZOOM}")
    print(f"{datetime.now().isoformat()} - collected Gardiner Expy traffic data!")
    gardiner_expy_content = gardiner_expy_response.content
    gardiner_expy_content_json = json.loads(gardiner_expy_content)
    gardiner_expy_content_json["flowSegmentData"]["location"] = TOMTOM_GARDINER_POINT
    gardiner_expy_content_json["flowSegmentData"]["collected_at"] = curr_datetime.isoformat(
    )
    del(gardiner_expy_content_json["flowSegmentData"]["coordinates"])

    print(f"{datetime.now().isoformat()} - collecting Front@Yonge traffic data...")
    front_yonge_response = r.get(
        f"{TOMTOM_BASE_URL}?key={TOMTOM_API_KEY}&point={TOMTOM_FRONT_YONGE_POINT}&{ZOOM}")
    print(f"{datetime.now().isoformat()} - collected Front@Yonge traffic data!")
    front_yonge_content = front_yonge_response.content
    front_yonge_content_json = json.loads(front_yonge_content)
    front_yonge_content_json["flowSegmentData"]["location"] = TOMTOM_FRONT_YONGE_POINT
    front_yonge_content_json["flowSegmentData"]["collected_at"] = curr_datetime.isoformat(
    )
    del(front_yonge_content_json["flowSegmentData"]["coordinates"])

    traffic_data_retrieved.append(gardiner_expy_content_json)
    traffic_data_retrieved.append(front_yonge_content_json)
    weather_data_retrieved.append(weather_content_json)

    requests_performed_per_day += 1
    print(f"{datetime.now().isoformat()} - got {requests_performed_per_day} requests so far!")

    if requests_performed_per_day == REQUESTS_PER_DAY:
        print(f"{datetime.now().isoformat()} - dumping traffic data to file...")
        with open(f"collected_data/{curr_datetime.isoformat()}_traffic.json", "w") as traffic_out_file:
            traffic_json_string = json.dumps(traffic_data_retrieved)
            traffic_out_file.write(traffic_json_string)
        print(f"{datetime.now().isoformat()} - dumped traffic data to file!")

        print(f"{datetime.now().isoformat()} - dumping weather data to file...")
        with open(f"collected_data/{curr_datetime.isoformat()}_weather.json", "w") as weather_out_file:
            weather_json_string = json.dumps(weather_data_retrieved)
            weather_out_file.write(weather_json_string)
        print(f"{datetime.now().isoformat()} - dumped weather data to file!")

        requests_performed_per_day = 0
        traffic_data_retrieved = list()
        weather_data_retrieved = list()

    print(f"{datetime.now().isoformat()} - sleeping for {SLEEP_TIME} seconds...")
    sleep(MINUTES_OF_SLEEP*MINUTE)

    curr_datetime = datetime.now()
