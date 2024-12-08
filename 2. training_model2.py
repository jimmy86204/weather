import os
import gc
import glob
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
from tqdm import tqdm

import sklearn
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

import joblib

## Set model paths
model_name = 'model2'
save_path = f'./output/{model_name}'
try:
    os.mkdir(save_path)
except:
    pass
try:
    os.mkdir(os.path.join(save_path, 'scaler'))
    os.mkdir(os.path.join(save_path, 'imputer'))
    os.mkdir(os.path.join(save_path, 'lstm'))
except:
    pass

## Define feature set
cols_emb = ['location_code', 'datetime_hour']
cols_feat_origin = ['windspeed', 'pressure', 'temperature', 'humidity', 'sunlight']
cols_feat_origin_timeslot = ['windspeed_timeavg', 'pressure_timeavg', 'temperature_timeavg', 'humidity_timeavg', 'sunlight_timeavg', 'power_timeavg']
cols_feat_opendata = ['pressure_open', 'temperature_open', 'rel_humidity_open', 'windspeed_open', 'precipitation_open', 'sunshine_duration_open']
cols_feat_scale = cols_feat_origin + cols_feat_opendata + ['power_scale']

cols_feat_all = cols_emb + cols_feat_origin + cols_feat_origin_timeslot + cols_feat_opendata + ['power_scale']
cols_feat_open = cols_emb + cols_feat_origin_timeslot + cols_feat_opendata

## Preprocessing methods
def prepare_features(data):
    ## rename columns
    data = data.sort_values(['LocationCode', 'DateTime'])\
               .rename(columns={"LocationCode"   : "location_code",
                                "DateTime"       : "datetime",
                                "WindSpeed(m/s)" : "windspeed",
                                "Pressure(hpa)"  : "pressure",
                                "Temperature(°C)": "temperature",
                                "Humidity(%)"    : "humidity",
                                "Sunlight(Lux)"  : "sunlight",
                                "Power(mW)"      : "power"})\
                .reset_index(drop=True)

    ## add datetime columns
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['datetime_date'] = data['datetime'].dt.date
    data['datetime_hour'] = data['datetime'].dt.hour
    data['datetime_10mins'] = data['datetime'].dt.floor('10min')
    # data['datetime_10mins'] = data['datetime'].dt.round('min').dt.floor('10min') ## avoid over 10 records in 10 mins

    ## add index to raw data
    data['index_dt']  = data['datetime_10mins'].dt.strftime('%Y%m%d%H%M')
    data['index_loc'] = data['location_code'].apply(lambda x: f"{x:02}")
    data['index'] = (data['index_dt'] + data['index_loc']).astype(np.int64)
    data = data.drop(columns=['index_dt', 'index_loc'])

    ## aggregate data
    cols_key = ['index', 'location_code', 'datetime_10mins', 'datetime_date', 'datetime_hour']
    cols_feat = ['windspeed', 'pressure', 'temperature', 'humidity', 'sunlight', 'power']

    data = data.groupby(cols_key)[cols_feat].mean().reset_index()
    data['is_train'] = 1

    return data

def prepare_targets(data):
    ## rename columns
    data = data.rename(columns={'序號': 'index', '答案': 'power'}).astype({'index': str, 'power': float})

    ## parse index
    data['datetime_10mins'] = data['index'].apply(lambda x: pd.to_datetime(str(x)[:-2]))
    data['location_code'] = data['index'].apply(lambda x: int(str(x)[-2:]))

    ## add datetime columns
    data['datetime_date'] = data['datetime_10mins'].dt.date
    data['datetime_hour'] = data['datetime_10mins'].dt.hour

    cols = ['index', 'location_code', 'datetime_10mins', 'datetime_date', 'datetime_hour', 'power']
    data = data[cols]
    data['is_train'] = 0

    return data

def prepare_weather_data(data):
    ## select relevant data
    data = data[data['site_name']== '花蓮'].reset_index(drop=True)

    ## rename columns
    data = data.sort_values(['DataTime'])\
               .rename(columns={'DataTime': 'datetime',
                                'AirPressure': 'pressure_open',
                                'AirTemperature': 'temperature_open',
                                'RelativeHumidity': 'rel_humidity_open',
                                'WindSpeed': 'windspeed_open',
                                'Precipitation': 'precipitation_open',
                                'SunshineDuration': 'sunshine_duration_open'})\
               .reset_index(drop=True)

    ## add datetime columns
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['datetime_date'] = data['datetime'].dt.date
    data['datetime_hour'] = data['datetime'].dt.hour

    ## correct data
    data.loc[data['precipitation_open'] == 'T', 'precipitation_open'] = 0.1

    cols = [
        'datetime_date', 'datetime_hour', 'pressure_open', 'temperature_open',
        'rel_humidity_open', 'windspeed_open', 'precipitation_open', 'sunshine_duration_open'
    ]
    data = data.astype({col: float for col in data.columns if '_open' in col})

    return data[cols]

## Feature Engineering
def feature_engineering(data_features, data_pred, dat_weather, cols_feat=cols_feat_scale):
    ## merge features and weather data
    data = pd.concat([data_features, data_pred], ignore_index=True)

    ## add weather avg of all locations
    data_timeavg = data.groupby('datetime_10mins')[cols_feat_origin+['power']].mean()
    data_timeavg.columns = [col + '_timeavg' for col in data_timeavg.columns]
    data_timeavg = data_timeavg.reset_index()
    data = data.merge(data_timeavg, on='datetime_10mins', how='left')

    ## merge features and weather data
    data = data.merge(dat_weather, on=['datetime_date', 'datetime_hour'], how='left')

    # filled missing values in interested columns
    if data[cols_emb+cols_feat_opendata[:-1]].isna().sum().sum() == 0:
        cols_impute = cols_emb+cols_feat_origin_timeslot+cols_feat_opendata[:-1]
        print("Before filling:")
        print(data[cols_impute].isna().sum())

        imputer = KNNImputer(n_neighbors=5)
        data[cols_impute] = imputer.fit_transform(data[cols_impute])
        print("After filling:")
        print(data[cols_impute].isna().sum())
        joblib.dump(imputer, os.path.join(save_path, 'imputer/imputer_timeavg.pkl'))
    else:
        print("Not filled null values with KNNImputer !")

    ## apply MinMaxScaler
    for col in cols_feat:
        if col == 'power_scale':
            col_fit = 'power'
            col_new = col
        else:
            col_fit = col
            col_new = col

        scaler = MinMaxScaler()
        values_fit = data.loc[data['is_train']==1, [col_fit]]
        scaler.fit(values_fit)
        data[col_new] = scaler.transform(data[[col_fit]])

        if col_new+'_timeavg' in cols_feat_origin_timeslot:
            data[col_new+'_timeavg'] = scaler.transform(data[[col_fit+'_timeavg']].values)

        print(f"{col_fit:<22} | Fit: {len(values_fit):,d}, Transform: {len(data[col_new]):,d}")
        joblib.dump(scaler, os.path.join(save_path, f'scaler/scaler_{col_fit}.pkl'))

    return data

def prepare_lstm_data(df_key, df_feat, cols_feat=cols_feat_all, cols_miss=cols_feat_origin,
                      cols_targetdate=None):
    if cols_targetdate is None:
        cols_targetdate = [col for col in cols_feat if col != 'power_scale']

    lstm_idx = []
    lstm_data = []
    for i, row in tqdm(df_key.iterrows()):
        location = row['location_code']
        pred_date = row['datetime_date']
        last_date = row['datetime_date_last']

        # Extract input & target sequence
        prompt_df = df_feat[
            (df_feat['location_code'] == location) &
            (
                (df_feat['datetime_date'] == last_date) & (df_feat['datetime_hour'].between(7, 16)) |
                (df_feat['datetime_date'] == pred_date) & (df_feat['datetime_hour'].between(7, 8))
            )
        ].sort_values('datetime_10mins')

        target_df = df_feat[
            (df_feat['location_code'] == location) &
            (df_feat['datetime_date'] == pred_date) & (df_feat['datetime_hour'].between(9, 16))
        ].sort_values('datetime_10mins')
        target_df = target_df.drop(columns=cols_miss)

        target_df_fill = df_feat[
            (df_feat['location_code'] == location) &
            (df_feat['datetime_date'] == last_date) & (df_feat['datetime_hour'].between(9, 16))
        ].sort_values('datetime_10mins')

        target_df_fill = target_df_fill[['location_code', 'datetime_10mins'] + cols_miss]
        target_df_fill['datetime_10mins'] = target_df_fill['datetime_10mins'] + pd.to_timedelta(1, unit='D')
        target_df = target_df.merge(target_df_fill, on=['location_code', 'datetime_10mins'], how='left')

        # if i % 200 == 0:
        #     print(prompt_df.shape, prompt_df.head())
        #     print(target_df.shape, target_df.head())

        if len(prompt_df) == 72 and len(target_df) == 48:
            X_prompt = prompt_df[cols_feat].values
            X_target = target_df[cols_targetdate].values
            y = target_df['power_scale'].values ## 'power
            X_idx = prompt_df['index'].values
            y_idx = target_df['index'].values
            lstm_idx.append((X_idx, y_idx))
            lstm_data.append((X_prompt, X_target, y))

    return lstm_idx, lstm_data

## Custom dataset
class PowerDataset(Dataset):
    def __init__(self, lstm_idx, lstm_data):
        self.lstm_idx = lstm_idx
        self.lstm_data = lstm_data

    def __len__(self):
        return len(self.lstm_data)

    def __getitem__(self, idx):
        x_prompt, x_target, y = self.lstm_data[idx]
        x_prompt_idx, x_target_idx = self.lstm_idx[idx]

        # Convert data to tensors
        x_prompt = torch.tensor(x_prompt, dtype=torch.float32)
        x_target = torch.tensor(x_target, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        # Convert numpy arrays to tensors
        x_prompt_idx = torch.tensor(x_prompt_idx.astype(np.int64), dtype=torch.int64)
        x_target_idx = torch.tensor(x_target_idx.astype(np.int64), dtype=torch.int64)

        return x_prompt, x_target, y, x_prompt_idx, x_target_idx

## Custom model
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, loc_embedding_dim, loc_vocab_size, time_embedding_dim, time_vocab_size):
        super(Encoder, self).__init__()
        self.loc_embedding = nn.Embedding(loc_vocab_size, loc_embedding_dim)
        self.time_embedding = nn.Embedding(time_vocab_size, time_embedding_dim)
        self.lstm = nn.LSTM(input_size+loc_embedding_dim+time_embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        embedded_loc = torch.squeeze(self.loc_embedding(x[:, :, 0].to(torch.int64)))
        embedded_time = torch.squeeze(self.time_embedding(x[:, :, 1].to(torch.int64)))
        x = torch.cat((embedded_loc, embedded_time, x[:, :, 2:]), dim=2)
        out, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, loc_embedding_dim, loc_vocab_size, time_embedding_dim, time_vocab_size):
        super(Decoder, self).__init__()
        self.loc_embedding = nn.Embedding(loc_vocab_size, loc_embedding_dim)
        self.time_embedding = nn.Embedding(time_vocab_size, time_embedding_dim)
        self.lstm = nn.LSTM(input_size+loc_embedding_dim+time_embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, input_size+loc_embedding_dim+time_embedding_dim)

    def forward(self, x, hidden, cell):
        embedded_loc = torch.squeeze(self.loc_embedding(x[:, :, 0].to(torch.int64)))
        embedded_time = torch.squeeze(self.time_embedding(x[:, :, 1].to(torch.int64)))
        x = torch.cat((embedded_loc, embedded_time, x[:, :, 2:]), dim=2)
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
        out = self.fc(out)
        return out, hidden, cell

class EncoderDecoderModel(nn.Module):
    def __init__(self, encoder_input_size, encoder_hidden_size, encoder_num_layers,
                       decoder_input_size, decoder_hidden_size, decoder_num_layers,
                       loc_embedding_dim, loc_vocab_size, time_embedding_dim, time_vocab_size, output_size):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = Encoder(encoder_input_size, encoder_hidden_size, encoder_num_layers,
                               loc_embedding_dim, loc_vocab_size, time_embedding_dim, time_vocab_size)
        self.decoder = Decoder(decoder_input_size, decoder_hidden_size, decoder_num_layers,
                               loc_embedding_dim, loc_vocab_size, time_embedding_dim, time_vocab_size)
        self.fc = nn.Linear(decoder_input_size+loc_embedding_dim+time_embedding_dim, output_size)

    def forward(self, encoder_input, decoder_input):
        hidden, cell = self.encoder(encoder_input)
        decoder_output, _, _ = self.decoder(decoder_input, hidden, cell)
        out = self.fc(decoder_output).squeeze(2)
        return out


if __name__ == "__main__":
    ## Load data
    paths_train_1 = glob.glob("./data/36_TrainingData/*.csv")
    paths_train_2 = glob.glob("./data/36_TrainingData_Additional_V2/*.csv")
    paths_test    = glob.glob("./data/36_TestSet_SubmissionTemplate/*.csv")
    paths_weather = "./data/open_weather_data.parq"

    data_train = pd.concat([pd.read_csv(path) for path in paths_train_1+paths_train_2], ignore_index=True)
    data_test  = pd.concat([pd.read_csv(path) for path in paths_test                 ], ignore_index=True)
    data_weather = pd.read_parquet(paths_weather)

    ## Preprocessing
    data_features = prepare_features(data_train)
    data_pred = prepare_targets(data_test)
    data_weather = prepare_weather_data(data_weather)

    ## Feature Engineering
    df_feat = feature_engineering(data_features, data_pred, data_weather)

    ## Generating valid training data
    df_quality = df_feat.loc[df_feat['is_train']==1, ['index', 'location_code', 'datetime_date', 'datetime_hour', 'power']]

    df_quality_lastday = df_quality.loc[df_quality['datetime_hour'].isin([7,8,9,10,11,12,13,14,15,16])].drop_duplicates()
    df_quality_lastday = df_quality_lastday.groupby(['location_code', 'datetime_date'])['power'].count().reset_index()
    df_quality_lastday = df_quality_lastday.loc[df_quality_lastday['power']==60]
    df_quality_lastday['key_date'] = df_quality_lastday['datetime_date'] + pd.to_timedelta(1, unit='D')
    df_quality_lastday['checked_lastday'] = 1

    df_quality_morning = df_quality.loc[df_quality['datetime_hour'].isin([7,8])].drop_duplicates()
    df_quality_morning = df_quality_morning.groupby(['location_code', 'datetime_date'])['power'].count().reset_index()
    df_quality_morning = df_quality_morning.loc[df_quality_morning['power']==12]
    df_quality_morning['key_date'] = df_quality_morning['datetime_date']
    df_quality_morning['checked_morning'] = 1

    df_quality_excl = df_feat.loc[df_feat['is_train']==0][['location_code', 'datetime_date']].drop_duplicates()
    df_quality_excl['to_exclude'] = 1

    df_quality_checked = df_quality_lastday[['location_code', 'key_date', 'checked_lastday']].merge(
        df_quality_morning[['location_code', 'key_date', 'checked_morning']],
        on=['location_code', 'key_date'], how='outer')
    df_quality_checked['checked'] = ( (df_quality_checked['checked_lastday']==1) & (df_quality_checked['checked_morning']==1) ).astype(int)
    df_quality_checked = df_quality_checked.loc[df_quality_checked['checked'] == 1, ['location_code', 'key_date']].rename(columns={'key_date': 'datetime_date'})
    df_quality_checked['datetime_date_last'] = df_quality_checked['datetime_date'] - pd.to_timedelta(1, unit='D')
    df_quality_checked = df_quality_checked.merge(df_quality_excl, on=['location_code', 'datetime_date'], how='left')
    df_quality_checked = df_quality_checked.loc[df_quality_checked['to_exclude'].isna()].reset_index(drop=True)
    del df_quality, df_quality_lastday, df_quality_morning, df_quality_excl
    gc.collect()

    ## Preparing LSTM training data
    lstm_idx, lstm_data = prepare_lstm_data(df_quality_checked, df_feat, cols_feat=cols_feat_all, cols_targetdate=cols_feat_open)

    ## Training
    ### hyperparameters
    encoder_input_size = 18
    encoder_hidden_size = 64
    encoder_num_layers = 3

    decoder_input_size = 12
    decoder_hidden_size = 64
    decoder_num_layers = 3

    output_size = 1

    loc_embedding_dim = 4
    loc_vocab_size = 18  # location_code 1-17 , no zero
    time_embedding_dim = 4
    time_vocab_size = 24

    learning_rate = 0.0003
    num_epochs = 300
    batch_size = 32
    es_tolerance = 20

    n_fold = 3

    ### cross validation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.L1Loss()

    #### prepare KFold
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)

    cv_folds_train = dict()
    cv_folds_val = dict()
    #### training loop with n-fold cross-validation
    for fold, (train_index, val_index) in enumerate(kf.split(lstm_data)):
        print(f"Fold {fold + 1}/{n_fold}")
        train_idx = [lstm_idx[i] for i in train_index]
        train_data = [lstm_data[i] for i in train_index]
        val_idx = [lstm_idx[i] for i in val_index]
        val_data = [lstm_data[i] for i in val_index]

        #### reinitialize model for each fold (important for proper cross-validation)
        model = EncoderDecoderModel(encoder_input_size, encoder_hidden_size, encoder_num_layers,
                                    decoder_input_size, decoder_hidden_size, decoder_num_layers,
                                    loc_embedding_dim, loc_vocab_size, time_embedding_dim, time_vocab_size, output_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        #### create DataLoaders
        train_dataset = PowerDataset(train_idx, train_data)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = PowerDataset(val_idx, val_data)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        #### training loop for the current fold
        epoch_train_losses = []
        epoch_val_losses = []
        es_counter = 0
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0
            cv_indexes = []
            cv_predictions = []
            for i, (encoder_input, decoder_input, targets, inputs_idx, targets_idx) in enumerate(train_dataloader):
                encoder_input = encoder_input.to(device)
                decoder_input = decoder_input.to(device)
                targets = targets.to(device)
                # forward pass
                outputs = model(encoder_input, decoder_input)
                # calculate Loss
                train_loss = criterion(outputs, targets)
                epoch_train_loss = np.nansum([epoch_train_loss, train_loss.item()*len(targets)])
                # backward and optimize
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                cv_indexes.append(targets_idx)
                cv_predictions.append(outputs.cpu().detach().numpy() if device != 'cpu' else outputs.detach().numpy())

            cv_indexes = np.concatenate(cv_indexes, axis=0)
            cv_predictions = np.concatenate(cv_predictions, axis=0)
            cv_folds_train[fold+1] = {'index': np.array(cv_indexes).reshape(-1),
                                    'prediction': np.array(cv_predictions).reshape(-1)}

            #### validation
            model.eval()
            epoch_val_loss = 0
            cv_indexes = []
            cv_predictions = []
            with torch.no_grad():
                for encoder_input, decoder_input, targets, inputs_idx, targets_idx in val_dataloader:
                    encoder_input = encoder_input.to(device)
                    decoder_input = decoder_input.to(device)
                    targets = targets.to(device)
                    # forward pass
                    outputs = model(encoder_input, decoder_input)
                    # calculate Loss
                    val_loss = criterion(outputs, targets)
                    epoch_val_loss = np.nansum([epoch_val_loss, val_loss.item()*len(targets)])

                    cv_indexes.append(targets_idx)
                    cv_predictions.append(outputs.cpu().detach().numpy() if device != 'cpu' else outputs.detach().numpy())

                cv_indexes = np.concatenate(cv_indexes, axis=0)
                cv_predictions = np.concatenate(cv_predictions, axis=0)
                cv_folds_val[fold+1] = {'index': np.array(cv_indexes).reshape(-1),
                                        'prediction': np.array(cv_predictions).reshape(-1)}

            epoch_train_loss /= len(train_dataset)
            epoch_val_loss /= len(val_dataset)

            epoch_train_losses.append(epoch_train_loss)
            epoch_val_losses.append(epoch_val_loss)

            if epoch % 10 == 9:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4e}, Val Loss: {epoch_val_loss:.4e}")

            #### early stopping check
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                es_counter = 0
                # save the best model (optional)
                torch.save(model.state_dict(), os.path.join(save_path, f'lstm/model_fold_{fold+1}.pth'))
            else:
                es_counter += 1
                if es_counter >= es_tolerance:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        #### display training loss
        # plt.figure(figsize=(10, 4))
        # plt.plot(epoch_train_losses, label='train')
        # plt.plot(epoch_val_losses, label='valid')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title(f'Training Loss @ Fold {fold+1}')
        # plt.legend()
        # plt.show()

        #### clear memory after each fold
        del model, optimizer, train_dataset, train_dataloader, val_dataset, val_dataloader
        gc.collect()
        torch.cuda.empty_cache()

    ## Generate prediction on training data and oof data
    ### train
    cols_pred = []
    for i, k in enumerate(cv_folds_train.keys()):
        col = 'pred_'+str(k)
        if i == 0:
            df_cv_train = pd.DataFrame(cv_folds_train[k]).rename(columns={'prediction': col})
        else:
            df_cv_train = df_cv_train.merge(pd.DataFrame(cv_folds_train[k]).rename(columns={'prediction': col}),
                                on='index', how='outer')
        cols_pred.append(col)
    df_cv_train['power'] = df_cv_train[cols_pred].mean(axis=1)
    df_cv_train['set'] = 'train'

    ### valid
    cols_pred = []
    for i, k in enumerate(cv_folds_val.keys()):
        col = 'pred_'+str(k)
        if i == 0:
            df_cv_val = pd.DataFrame(cv_folds_val[k]).rename(columns={'prediction': col})
        else:
            df_cv_val = df_cv_val.merge(pd.DataFrame(cv_folds_val[k]).rename(columns={'prediction': col}),
                                on='index', how='outer')
        cols_pred.append(col)
    df_cv_val['power'] = df_cv_val[cols_pred].mean(axis=1)
    df_cv_val['set'] = 'val'

    ### merge
    df_cv = pd.concat([df_cv_train, df_cv_val])
    del df_cv_train, df_cv_val
    gc.collect()

    ## Inference
    df_pred_checked = data_pred[['location_code', 'datetime_date']].drop_duplicates().reset_index(drop=True)
    df_pred_checked['datetime_date_last'] = df_pred_checked['datetime_date'] - pd.to_timedelta(1, unit='D')
    
    ### preparing LSTM testing data
    lstm_idx, lstm_data = prepare_lstm_data(df_pred_checked, df_feat, cols_feat=cols_feat_all, cols_targetdate=cols_feat_open)

    ### submit
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preds_fold = dict()
    encoder_input = torch.tensor([data[0] for data in lstm_data], dtype=torch.float32).to(device)
    decoder_input = torch.tensor([data[1] for data in lstm_data], dtype=torch.float32).to(device)
    targets_idx = [idx[1] for idx in lstm_idx]
    for fold in range(n_fold):
        path_model = os.path.join(save_path, f'lstm/model_fold_{fold+1}.pth')
        model = EncoderDecoderModel(encoder_input_size, encoder_hidden_size, encoder_num_layers,
                                    decoder_input_size, decoder_hidden_size, decoder_num_layers,
                                    loc_embedding_dim, loc_vocab_size, time_embedding_dim, time_vocab_size, output_size).to(device)
        model.load_state_dict(torch.load(path_model))
        model.eval()

        preds = model(encoder_input, decoder_input)
        if device != 'cpu':
            preds = preds.cpu()
        preds = preds.detach().numpy()
        preds_fold[fold+1] = {'index': np.array(targets_idx).reshape(-1), 'prediction': np.array(preds).reshape(-1)}

    cols_pred = []
    for i, k in enumerate(preds_fold.keys()):
        col = 'pred_'+str(k)
        if i == 0:
            df_sub = pd.DataFrame(preds_fold[k]).rename(columns={'prediction': col})
        else:
            df_sub = df_sub.merge(pd.DataFrame(preds_fold[k]).rename(columns={'prediction': col}), on='index', how='left')
        cols_pred.append(col)
    df_sub['power'] = df_sub[cols_pred].mean(axis=1)

    ## Dump
    df_sub_ = df_sub.copy()
    df_sub_['set'] = 'test'
    df_eval = pd.concat([df_cv, df_sub_])
    
    scaler = joblib.load(os.path.join(save_path, 'scaler/scaler_power.pkl'))
    for col in ['pred_1', 'pred_2', 'pred_3', 'power']:
        df_eval[col] = scaler.inverse_transform(df_eval[col].values.reshape(-1, 1)).reshape(-1)
        df_eval[col] = df_eval[col].clip(lower=0)
    
    df_sub['power'] = scaler.inverse_transform(df_sub['power'].values.reshape(-1, 1)).reshape(-1)
    df_sub['power'] = df_sub['power'].clip(lower=0)
    df_sub = df_sub[['index', 'power']]
    df_sub = df_sub.rename(columns={'index': '序號', 'power': '答案'})

    df_eval.to_csv(os.path.join(save_path, 'oof.csv'), index=False)
    df_sub.to_csv(os.path.join(save_path, 'sub.csv'), index=False)
    
