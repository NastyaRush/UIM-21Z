from pathlib import Path
import pandas as pd
import os
import sys


def write_processed_data(data, path):
    data = pd.DataFrame(data, columns = ['sessions_time_sum', 'greater_than_mean_buy_event_number'])
    return data.to_json(fr'{path}input_data.jsonl', orient='records')


def read_data(actual_path=None, path=None):
    # read processed data for models from processed dir
    if actual_path is None and path is None:
        actual_path = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")
        data_dir = str(actual_path).replace('/project_name/data', '/data/processed/')
        data = pd.read_json(Path(data_dir + '/input_data.jsonl'), orient='records')
        return data.to_numpy()
    # read raw data
    data_dir = str(actual_path).replace('/project_name/data', path)
    data = pd.read_json(Path(data_dir + '/sessions.jsonl'), lines=True)
    return data


def main():
    # find actual path
    actual_path = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")
    df_sessions = read_data(actual_path, '/data/raw/')
    # get path to build_features
    sys.path.insert(1, actual_path.replace('data', 'features'))
    import build_features
    # processing data
    processed_data = build_features.preprocessing_data(df_sessions)
    # write processed data to folder
    write_processed_data(processed_data, str(actual_path).replace('/project_name/data', '/data/processed/'))

if __name__ == '__main__':
     main()
