import pandas as pd
import os
import numpy as np
import glob
import gc
import datetime

from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from lightgbm import LGBMRegressor
import joblib
import pickle
import re
from sklearn.pipeline import Pipeline
from optuna.integration import LightGBMPruningCallback
import catboost
from catboost import CatBoostRegressor

def round_down_to_previous_10_minutes(dt):
    # 減去多餘的分鐘數和秒數
    minutes_to_subtract = dt.minute % 10
    rounded_dt = dt - datetime.timedelta(minutes=minutes_to_subtract, 
                                        seconds=dt.second, 
                                        microseconds=dt.microsecond)
    return rounded_dt

def create_early_features(df_, col):
    Sunlight_mean = df_.groupby(['Datetime_hour', 'LocationCode'])[col].mean().reset_index()
    Sunlight_mean['date'] = Sunlight_mean.Datetime_hour.dt.date
    Sunlight_mean['hour'] = Sunlight_mean.Datetime_hour.dt.hour
    Sunlight_mean = Sunlight_mean[Sunlight_mean.hour <= 8].reset_index(drop=True)
    Sunlight_mean = Sunlight_mean.sort_values(by=['Datetime_hour', 'LocationCode']).reset_index(drop=True)
    Sunlight_mean[f'{col}_prev1'] = Sunlight_mean.groupby(['LocationCode']).shift(1)[col]
    Sunlight_mean[f'{col}_prev2'] = Sunlight_mean.groupby(['LocationCode']).shift(2)[col]
    
    Sunlight_mean = Sunlight_mean[Sunlight_mean.hour == 8].reset_index(drop=True)
    Sunlight_mean[f'{col}_change1'] = Sunlight_mean[col] / (Sunlight_mean[f'{col}_prev1'] + 1e-4)
    Sunlight_mean[f'{col}_change2'] = Sunlight_mean[col] / (Sunlight_mean[f'{col}_prev2'] + 1e-4)
    use_col = [col, f'{col}_change1', f'{col}_change2']
    return Sunlight_mean[use_col+['date', 'LocationCode']], [], use_col

def fe(df_):
    df = df_.copy()
    df['date'] = df.DateTime.dt.date
    df['hour'] = df.DateTime.dt.hour
    df['minutes'] = df.DateTime.dt.minute
    df['weekday'] = df.DateTime.dt.weekday
    df['month'] = df.DateTime.dt.month
    df = df.merge(location_detail_df, on=['LocationCode'], how='left')
    
    original_features = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Sunlight(Lux)']
    tmp = concat_df.groupby(['DateTime'])[original_features].mean().reset_index().rename(columns={x : f"{x}_mean" for x in original_features})
    use_cols = [f"{x}_mean" for x in original_features]
    tmp = pd.concat([tmp, tmp.shift(1)[use_cols].add_suffix('_shift1'), tmp.shift(-1)[use_cols].add_suffix('_shift-1')], axis=1)
    for feature in original_features:
        tmp[f'{feature}_mean_shift1_ratio'] = tmp[f'{feature}_mean'] / (tmp[f'{feature}_mean_shift1'] + 1e-4)
        tmp[f'{feature}_mean_shift-1_ratio'] = tmp[f'{feature}_mean'] / (tmp[f'{feature}_mean_shift-1'] + 1e-4)
        use_cols += [f'{feature}_mean_shift1_ratio', f'{feature}_mean_shift-1_ratio']
    df = df.merge(
        tmp, on=['DateTime'], how='left'
    )
    
    early_features_list = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Sunlight(Lux)']
    cat_cols = ['hour', 'month', 'minutes', 'weekday', 'WindDirection', 'Direction', 'Level']
    num_cols = f + ['Latitude', 'Longitude', 'Degree'] + use_cols
    for col in early_features_list:
        res, cat_col, num_col = create_early_features(df, col)
        df = df.drop(columns=[col]).merge(res, on=['date', 'LocationCode'], how='left')
        num_cols += num_col
    return df, cat_cols, num_cols

def trainer(train_x, train_y, valid_x, valid_y, cat_cols):
    lgb_params = {
        'objective' : 'regression',
        'metric' : 'mae',
        'learning_rate': 0.05,
        'max_depth': 8,
        'num_leaves': 64,

        'num_iterations': 55000,
        'boosting_type': 'gbdt', # dart
        'seed': 42,

        'feature_fraction': 0.75,
        'bagging_freq': 5,
        'bagging_fraction': 0.50,
        'max_bin': 100,
        'min_data_in_leaf': 20,

        'n_jobs': 8,
        'verbose': -1,
        'lambda_l2': 2,
    }
    es = lgb.early_stopping(1000, verbose=False, min_delta=1e-2)
    log = lgb.log_evaluation(period=100)
    reg = LGBMRegressor(**lgb_params)
    reg.fit(train_x,
            train_y,
            categorical_feature=cat_cols,
            eval_set = [(train_x, train_y),
                        (valid_x, valid_y)],
            callbacks=[log, es],
               )
    return reg

if __name__ == '__main__':
    location_detail = [[1, 23.899444, 121.544444, 181, "S", 5],
                       [2, 23.899722, 121.544722, 175, "S", 5],
                       [3, 23.899722, 121.545000, 180, "S", 5],
                       [4, 23.899444, 121.544444, 161, "S", 5],
                       [5, 23.899444, 121.544722, 208, "WS", 5],
                       [6, 23.899444, 121.544444, 208, "WS", 5],
                       [7, 23.899444, 121.544444, 172, "S", 5],
                       [8, 23.899722, 121.545000, 219, "WS", 3],
                       [9, 23.899444, 121.544444, 151, "ES", 3],
                       [10, 23.899444, 121.544444, 223, "WS", 1],
                       [11, 23.899722, 121.544722, 131, "ES", 1],
                       [12, 23.899722, 121.544722, 298, "WN", 1],
                       [13, 23.897778, 121.539444, 249, "W", 5],
                       [14, 23.897778, 121.539444, 197, "S", 5],
                       [15, 24.009167, 121.617222, 127, "ES", 1],
                       [16, 24.008889, 121.617222, 82, 'E', 1],
                       [17, 23.986557, 121.586230, np.nan, np.nan, np.nan],]
    location_detail_df = pd.DataFrame(location_detail, columns=['LocationCode', 'Latitude', 'Longitude', 'Degree', "Direction", "Level"])
    csv_files_train_1 = glob.glob("./data/36_TrainingData/*.csv")
    csv_files_train_2 = glob.glob("./data/36_TrainingData_Additional_V2/*.csv")
    csv_files_test_1 = glob.glob("./data/36_TestSet_SubmissionTemplate/*.csv")
    dataframes = [pd.read_csv(file) for file in csv_files_train_1 + csv_files_train_2]
    train_df = pd.concat(dataframes, ignore_index=True)
    dataframes = [pd.read_csv(file) for file in csv_files_test_1]
    test_df = pd.concat(dataframes, ignore_index=True)
    test_df['DateTime'] = test_df['序號'].apply(lambda x: pd.to_datetime(str(x)[:-2]))
    test_df['LocationCode'] = test_df['序號'].apply(lambda x: int(str(x)[-2:]))
    test_df['Power(mW)'] = None
    test_df['is_train'] = 0
    del test_df['序號'], test_df['答案']
    gc.collect()
    train_df['DateTime'] = pd.to_datetime(train_df.DateTime)
    train_df.DateTime = train_df.DateTime.apply(round_down_to_previous_10_minutes)
    train_df = train_df.groupby(['LocationCode', 'DateTime'])[["WindSpeed(m/s)", "Pressure(hpa)", "Temperature(°C)", "Sunlight(Lux)", "Power(mW)"]].mean().reset_index()
    train_df['is_train'] = 1
    concat_df = pd.concat([train_df, test_df], ignore_index=True)
    concat_df = concat_df.sort_values(by=['LocationCode', 'DateTime']).reset_index(drop=True)
    concat_df['Datetime_hour'] = pd.to_datetime(concat_df.DateTime.dt.strftime('%Y-%m-%d %H:00:00'))

    df_weather = pd.read_parquet('./data/open_weather_data.parq')
    df_weather_sub = df_weather[df_weather.site_name == '花蓮'].reset_index(drop=True)
    df_weather_sub.loc[df_weather_sub.Precipitation == 'T', 'Precipitation'] = 0.1
    f = ['AirPressure', 'AirTemperature', 'RelativeHumidity', 'WindSpeed', 'Precipitation', 'SunshineDuration']
    for col in f:
        df_weather_sub[col] = df_weather_sub[col].astype(float)
    df_weather_sub = pd.concat([df_weather_sub, 
                                df_weather_sub.shift(1)[f].add_suffix('_nextState1'), 
                                df_weather_sub.shift(2)[f].add_suffix('_nextState2'), 
                                df_weather_sub.shift(-1)[f].add_suffix('_prevState1'),
                                df_weather_sub.shift(-2)[f].add_suffix('_prevState2'),
                                ], axis=1)
    add_d = []
    for col in f:
        df_weather_sub[f'{col}_nextState_change1'] = df_weather_sub[col] / (df_weather_sub[col + '_nextState1'] + 1e-4)
        df_weather_sub[f'{col}_prevState_change1'] = df_weather_sub[col] / (df_weather_sub[col + '_prevState1'] + 1e-4)
        df_weather_sub[f'{col}_nextState_change2'] = df_weather_sub[col] / (df_weather_sub[col + '_nextState2'] + 1e-4)
        df_weather_sub[f'{col}_prevState_change2'] = df_weather_sub[col] / (df_weather_sub[col + '_prevState2'] + 1e-4)
        add_d += [f'{col}_nextState_change1', f'{col}_prevState_change1', f'{col}_nextState_change2', f'{col}_prevState_change2']
    f += add_d
    concat_df = concat_df.merge(
        df_weather_sub.rename(columns={'DataTime': "Datetime_hour"}).drop(columns=['site_name', 'site_id', 'hour']),
        on=['Datetime_hour'], how='left'
    )
    concat_fe_df, cat_cols, num_cols = fe(concat_df)
    features = cat_cols + num_cols
    training_df = concat_fe_df[concat_fe_df.is_train == 1].reset_index(drop=True)
    testing_df = concat_fe_df[concat_fe_df.is_train == 0].reset_index(drop=True)
    le_dict = {}
    for i, col in tqdm(enumerate(cat_cols)):
        le = LabelEncoder()
        training_df[col] = le.fit_transform(training_df[col])
        le_dict[col] = le
    n_splits = 5
    training_df['fold'] = -1
    y = 'Power(mW)'
    sgkf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    for fold, (train_index, valid_index) in enumerate(sgkf.split(training_df['fold'])):
        training_df.loc[valid_index, 'fold'] = fold
    model_name = 'model1'
    save_path = f'./output/{model_name}'
    try:
        os.mkdir(save_path)
    except:
        pass
    oof_predictions = pd.DataFrame()
    all_result_record = []
    for fold in range(n_splits):
        print(f"staring training {fold}")
        training = training_df[training_df.fold != fold].reset_index(drop=True)
        validation = training_df[training_df.fold == fold].reset_index(drop=True)
    
        print('*'*30)
        print("num of features:", len(features))
        reg = trainer(training[features], training[y], validation[features], validation[y], cat_cols)
        all_result_record.append(reg)
        preds = reg.predict(validation[features])
        validation["preds"] = preds
        validation["fold"] = fold
        oof_predictions = pd.concat([oof_predictions, validation[["preds", "LocationCode", y, "fold", "DateTime"]]], axis=0)
    oof_predictions.to_parquet(os.path.join(save_path, "oof.parq"))
    for i, (col, le) in tqdm(enumerate(le_dict.items())):
        testing_df[col] = le.transform(testing_df[col])
    pred_cols = []
    for i, model in enumerate(all_result_record):
        testing_df[f'preds_{i}'] = model.predict(testing_df[features])
        pred_cols.append(f'preds_{i}')
    testing_df['preds'] = testing_df[pred_cols].mean(axis=1)
    if 'LocationCode' in cat_cols:
        testing_df['LocationCode'] = le_dict['LocationCode'].inverse_transform(testing_df.LocationCode)
    sub = pd.read_csv("./data/36_TestSet_SubmissionTemplate/upload(no answer).csv")
    testing_df['序號'] = (testing_df.DateTime.dt.strftime('%Y%m%d%H%M') + testing_df.LocationCode.apply(lambda x: str(x).zfill(2))).astype(np.int64)
    sub = sub.merge(
        testing_df[['序號', 'preds']], on=['序號'], how='left'
    )
    del sub['答案']
    sub = sub.rename(columns={'preds': '答案'})
    sub['答案'] = sub['答案'].clip(0, None).round(2)
    sub.to_csv(os.path.join(save_path, 'sub.csv'), index=False)