import datetime
import logging
import os
import requests as r
import pytz
import json

import azure.functions as func


def main(mytimer: func.TimerRequest, outputBlob: func.Out[str]) -> None:
    utc_timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc).isoformat()

    if mytimer.past_due:
        logging.info('The timer is past due!')

    logging.info('Python timer trigger function ran at %s', utc_timestamp)

    TOMTOM_API_KEY = os.getenv('TOMTOM_API_KEY')
    OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
    OPENWEATHER_BASE_URL = os.getenv('OPENWEATHER_BASE_URL')
    OPENWEATHER_COORD = os.getenv('OPENWEATHER_COORD')
    TRAFFIC_BASE_URL = os.getenv('TRAFFIC_BASE_URL')
    GARDINER_EXPY_POINT = os.getenv('GARDINER_EXPY_POINT')
    FRONT_YONGE_POINT = os.getenv('FRONT_YONGE_POINT')
    INCIDENTS_BASE_URL = os.getenv('INCIDENTS_BASE_URL')
    INCIDENTS_BOUNDING_BOX = os.getenv('INCIDENTS_BOUNDING_BOX')
    INCIDENTS_TIME_VALIDITY_FILTER = os.getenv(
        'INCIDENTS_TIME_VALIDITY_FILTER')
    INCIDENTS_FIELDS = os.getenv('INCIDENTS_FIELDS')
    INCIDENTS_LANG = os.getenv('INCIDENTS_LANG')

    collected_at = datetime.datetime.now(
        pytz.timezone('Canada/Eastern')).isoformat()

    logging.info('Collecting data...')
    gardiner_response = r.get(
        f"{TRAFFIC_BASE_URL}?key={TOMTOM_API_KEY}&point={GARDINER_EXPY_POINT}&zoom=22")
    front_yonge_response = r.get(
        f"{TRAFFIC_BASE_URL}?key={TOMTOM_API_KEY}&point={FRONT_YONGE_POINT}&zoom=22")
    incidents_response = r.get(
        f"{INCIDENTS_BASE_URL}?key={TOMTOM_API_KEY}&bbox={INCIDENTS_BOUNDING_BOX}&timeValidityFilter={INCIDENTS_TIME_VALIDITY_FILTER}&fields={INCIDENTS_FIELDS}&language={INCIDENTS_LANG}")
    weather_info_response = r.get(
        f"{OPENWEATHER_BASE_URL}&{OPENWEATHER_COORD}&appid={OPENWEATHER_API_KEY}")
    logging.info('Collecting data... OK')

    if gardiner_response.ok and front_yonge_response.ok and incidents_response.ok and weather_info_response.ok:
        gardiner_content = gardiner_response.content
        front_yonge_content = front_yonge_response.content
        incidents_content = incidents_response.content
        weather_info_content = weather_info_response.content

        gardiner_json = json.loads(gardiner_content)
        front_yonge_json = json.loads(front_yonge_content)
        incidents_json = json.loads(incidents_content)
        weather_info_json = json.loads(weather_info_content)

        info_json = {'collected_at': collected_at,
                     'gardiner_expy_traffic': gardiner_json,
                     'front_yonge_traffic': front_yonge_json,
                     'incidents': incidents_json,
                     'weather_info': weather_info_json}

        info_str = json.dumps(info_json)

        logging.info('Saving data to blob...')
        outputBlob.set(info_str)
        logging.info('Saving data to blob... OK')
    else:
        logging.info('Gardiner Expy status code: %s',
                     gardiner_response.status_code)
        logging.info('Front @ Yonge status code: %s',
                     front_yonge_response.status_code)
        logging.info('Incidents status code: %s',
                     incidents_response.status_code)
        logging.info('Weather API status code: %s',
                     weather_info_response.status_code)
