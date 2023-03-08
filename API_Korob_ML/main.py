import json

import dill

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

import warnings
warnings.filterwarnings('ignore')

app = FastAPI()
with open('target_action_predict.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    session_id: str = ''
    client_id: str = ''
    utm_source: str = ''
    utm_medium: str = ''
    utm_campaign: str = ''
    utm_adcontent: str = ''
    utm_keyword: str = ''
    device_category: str = ''
    device_os: str = ''
    device_brand: str = ''
    device_model: str = ''
    device_screen_resolution: str = ''
    device_browser: str = ''
    geo_country: str = ''
    geo_city: str = ''


class Prediction(BaseModel):
    client_id: str
    proba: float
    result: bool


@app.get('/')
def status():
    return "Let's START"


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    print('======= started ==========')

    df = pd.DataFrame.from_dict([form.dict()])

    client_id = df['client_id'][0]
    df = df_processing(df)
    proba = model['model'].predict_proba(df)[0][1]

    return {
        'client_id': client_id,
        'proba': proba,
        'result': proba > model['best_trsh']
    }


def df_processing(df):

    df = df[['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword', 'device_category', 'device_os',
             'device_brand',  'device_screen_resolution', 'device_browser', 'geo_country', 'geo_city']]

    df['utm_source'] = df['utm_source'].fillna(model['most_frequent_utm_source'])
    # пустые строки не воспринимаются как пропуски, добавляю для них отдельную обработку
    df.loc[(df.utm_keyword == ''), 'utm_keyword'] = model['most_frequent_utm_source']

    df[model['col_for_empty']].fillna('empty', inplace=True)
    # пустые строки не воспринимаются как пропуски, добавляю для них отдельную обработку
    for cl in model['col_for_empty']:
        df.loc[(df[cl] == ''), cl] = 'empty'

    # Если device_browser == 'Safari' заполняем пропуски значением 'Apple' (более 41 000 записей)
    df.loc[(df.device_brand == 'empty') & (df.device_browser == 'Safari'), 'device_brand'] = 'Apple'
    # Если device_os == 'Macintosh' заполняем пропуски значением 'Apple' (более 11 318 записей)
    df.loc[(df.device_brand == 'empty') & (df.device_os == 'Macintosh'), 'device_brand'] = 'Apple'
    # Если device_browser == 'Samsung Internet' заполняем пропуски значением 'Samsung' (655 записей)
    df.loc[(df.device_brand == 'empty') & (df.device_browser == 'Samsung Internet'), 'device_brand'] = 'Samsung'

    # Если device_brand == 'Apple' заполняем пропуски значением 'iOS'
    df.loc[(df.device_os == 'empty') & (df.device_brand == 'Apple'), 'device_os'] = 'iOS'
    # Если device_category.isin(['mobile', 'tablet']) заполняем пропуски значением 'Android'
    df.loc[(df.device_os == 'empty') & (df.device_category.isin(['mobile', 'tablet'])), 'device_os'] = 'Android'
    # Если device_category == 'desktop' заполняем пропуски значением 'Windows'
    df.loc[(df.device_os == 'empty') & (df.device_category == 'desktop'), 'device_os'] = 'Windows'

    df['organic_traffic'] = df['utm_medium'].isin(['organic', 'referral', '(none)'])

    df['social_media_advertising'] = df['utm_source'].isin(
        ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs',
         'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm'])

    return df

