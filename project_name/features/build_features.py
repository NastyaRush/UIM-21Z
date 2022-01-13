import pandas as pd
import numpy as np
from sklearn.preprocessing import scale


def preprocessing_data(data):
    users_sessions = data[['session_id', 'user_id', 'timestamp']].dropna().drop_duplicates()
    return configure_data(users_sessions, data)

def configure_data(users_sessions, df_sessions):
    # group by users_id and calculate time of session
    users_sessions["timestamp_diff"] = users_sessions.groupby('session_id')["timestamp"].diff().apply(lambda x: x/np.timedelta64(1, 's')).fillna(0).astype('int64')
    # sum all users sessions time  
    sessions_time = users_sessions.groupby('user_id')["timestamp_diff"].sum()
    df_sessions_time = sessions_time.to_frame().reset_index()
    df_sessions_time.rename(columns={"timestamp_diff": "sessions_time_sum"}, inplace=True)
    # group by user_id and count(event_type == BUY_PRODUCT)
    buy_event_number = df_sessions.groupby('user_id')['event_type'].apply(lambda x: x[x == 'BUY_PRODUCT'].count())
    df_buy_event_number = buy_event_number.to_frame().reset_index()
    df_buy_event_number.rename(columns={"event_type": "buy_event_number"}, inplace=True)
    #
    input_data = pd.concat([df_sessions_time["sessions_time_sum"]], axis=1)
    #
    mean_buy_event_number = df_buy_event_number["buy_event_number"].mean()
    input_data['greater_than_mean_buy_event_number'] = df_buy_event_number["buy_event_number"].apply(lambda x: 1 if x > mean_buy_event_number else 0)
    input_data = scale(input_data)

    return input_data
