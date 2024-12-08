import json
from tqdm.auto import tqdm
import datetime
import pandas as pd

with open('./data/extra.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
res = data['cwaopendata']['resources']['resource']['data']['surfaceObs']['location']
site_list = []
site_id_list = []
datetime_list = []
AirPressure_list = []
AirTemperature_list = []
RelativeHumidity_list = []
WindSpeed_list = []
WindDirection_list = []
Precipitation_list = []
SunshineDuration_list = []
for obj in tqdm(res):
    site_name = obj['station']['StationName']
    site_id = obj['station']['StationID']
    for info in obj['stationObsTimes']['stationObsTime']:
        site_list.append(site_name)
        site_id_list.append(site_id)
        datetime_list.append(info['DataTime'])
        AirPressure_list.append(info['weatherElements']['AirPressure'])
        AirTemperature_list.append(info['weatherElements']['AirTemperature'])
        RelativeHumidity_list.append(info['weatherElements']['RelativeHumidity'])
        WindSpeed_list.append(info['weatherElements']['WindSpeed'])
        WindDirection_list.append(info['weatherElements']['WindDirection'])
        Precipitation_list.append(info['weatherElements']['Precipitation'])
        SunshineDuration_list.append(info['weatherElements']['SunshineDuration'])
res = pd.DataFrame({
    'site_name': site_list,
    'site_id': site_id_list,
    'DataTime': datetime_list,
    'AirPressure': AirPressure_list,
    'AirTemperature': AirTemperature_list,
    'RelativeHumidity': RelativeHumidity_list,
    'WindSpeed': WindSpeed_list,
    'WindDirection': WindDirection_list,
    'Precipitation': Precipitation_list,
    'SunshineDuration': SunshineDuration_list,
})
res.DataTime = res.DataTime.apply(lambda x: pd.to_datetime(x[:19]))
res['hour'] = res.DataTime.dt.hour
res.to_parquet('./data/open_weather_data.parq')