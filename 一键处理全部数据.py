import math
import pandas as pd
import transbigdata as tbd
import datetime
import json
import os

os.makedirs('./data_csv', exist_ok=True)
os.makedirs('./data_csv/run', exist_ok=True)
os.makedirs('./data_csv/stop', exist_ok=True)
os.makedirs('./data_csv/charged', exist_ok=True)
os.makedirs('./data_csv/potential', exist_ok=True)
os.makedirs('./data_json', exist_ok=True)
os.makedirs('./data_json/run', exist_ok=True)
os.makedirs('./data_json/stop', exist_ok=True)
os.makedirs('./data_json/charged', exist_ok=True)
os.makedirs('./data_json/potential', exist_ok=True)

pd.set_option('display.max_columns', 1000)
pd.set_option('display.min_rows', 50)
pd.set_option('expand_frame_repr', False)

for file in os.listdir('新数据'):
    origin = pd.read_csv('新数据/' + file, names=['car_id', 'time', 'car_state', 'charged_state', 'run_model', 'speed', 'sum_km', 'sum_V', 'sum_A', 'SOC', 'DC-DC', 'gear', 'kOhm', 'lon', 'lat', 'max_V', 'min_V', 'max_temp', 'min_temp', 'motor_seq', 'motor_speed', 'motor_tor'])

    # 数据清洗
    data = origin.drop(columns=['car_state', 'run_model', 'DC-DC', 'gear', 'kOhm', 'speed', 'sum_km', 'sum_V', 'sum_A', 'max_V', 'min_V', 'max_temp', 'min_temp', 'motor_seq', 'motor_speed', 'motor_tor'], axis=1)

    # 剔除范围外数据
    data = data[data['lon'] > 120]
    data = data[data['lon'] < 123]
    data = data[data['lat'] > 30]
    data = data[data['lat'] < 32]

    data['SOC_last'] = data['SOC'].shift()
    data = data[data['SOC'] - data['SOC_last'] < 2]
    data = data[data['SOC'] - data['SOC_last'] > -2]
    del data['SOC_last']

    # data['charged_state'] = data['charged_state'].apply(lambda _: 3 if _ == 2 else _)

    # car_id 设为简单的1，2，3……
    data['car_id_next'] = data['car_id'].shift()
    data['car_id_alter'] = data.apply(lambda _: _['car_id'] != _['car_id_next'], axis=1)
    data['_'] = data['car_id_alter'].cumsum()
    data.drop(['car_id', 'car_id_next', 'car_id_alter'], axis=1, inplace=True)
    data.rename(columns={'_': 'car_id'}, inplace=True)

    def date_trans(da):
        a = None
        try:
            a = datetime.datetime.strptime(da, '%Y-%m-%d %H:%M:%S')
        except:
            a = None
        finally:
            return a

    # 时间str转datetime
    data['time'] = data['time'].apply(lambda _: date_trans(_))

    # 计算 poschange 和 posid
    data['lon_last'] = data['lon'].shift()
    data['lat_last'] = data['lat'].shift()

    data['poschange'] = data.apply(lambda _: int(_['lon'] == _['lon_last'] and _['lat'] == _['lat_last']), axis=1)
    data['pcs'] = data['poschange'].shift(-1)
    data['abs'] = data.apply(lambda _: abs(_['pcs'] - _['poschange']), axis=1)
    data['posid'] = data.groupby(['car_id'])['abs'].cumsum()
    data.drop(['lon_last', 'lat_last', 'pcs', 'abs'], axis=1, inplace=True)
    data.dropna(inplace=True)

    # group by
    def count_dis_by_tbd(df):
        df['lon_last'] = df['lon'].shift()
        df['lat_last'] = df['lat'].shift()
        df.dropna(inplace=True)
        return int(sum([tbd.getdistance(__['lon'], __['lat'], __['lon_last'], __['lat_last']) for __ in df.iloc]))


    gb = data.groupby(['car_id', 'posid'])
    data2 = pd.concat([gb['time'].apply(lambda _: _.iloc[0]), gb['time'].apply(lambda _: _.iloc[-1]),
                       gb['SOC'].apply(lambda _: _.iloc[0]), gb['SOC'].apply(lambda _: _.iloc[-1]),
                       gb['lon'].apply(lambda _: _.iloc[0]), gb['lat'].apply(lambda _: _.iloc[0]),
                       gb['lon'].apply(lambda _: _.iloc[-1]), gb['lat'].apply(lambda _: _.iloc[-1]),
                       gb.apply(count_dis_by_tbd)], axis=1)
    data2.columns = ['time_start', 'time_end', 'SOC_start', 'SOC_end', 'lon_start', 'lat_start', 'lon_end', 'lat_end',
                     'dis_sum']
    data2.reset_index(inplace=True)

    # drop_duplicates
    data3 = data.drop_duplicates(subset=['car_id', 'posid'])
    data3 = data3.drop(['time', 'SOC', 'lon', 'lat'], axis=1)

    # merge
    data4 = pd.merge(left=data2, right=data3)

    min30 = pd.Timedelta(minutes=30)
    data666 = data4.copy()
    data666['new_poschange'] = data666.apply(lambda _: 0 if _['time_end'] - _['time_start'] < min30 else 1 - _['poschange'], axis=1)
    data666.rename(columns={'time_start': 'time'}, inplace=True)
    data666.drop(['poschange', 'posid', 'SOC_start', 'SOC_end', 'lon_start', 'lat_start', 'lon_end', 'lat_end', 'charged_state', 'dis_sum'], axis=1, inplace=True)

    data777 = pd.merge(data, data666, on=['car_id', 'time'], how='outer')

    last_new_poschange = data777['new_poschange'][0]
    last_time_end = data777['time_end'][0]
    # min1 = pd.Timedelta(minutes=1)

    def update(time, poschange, new_poschange, time_end):
        global last_new_poschange
        global last_time_end
        if not math.isnan(new_poschange):
            last_new_poschange = new_poschange
            last_time_end = time_end  # + min1
            return new_poschange
        elif time <= last_time_end:
            return last_new_poschange
        else:
            return poschange


    data888 = data777.copy()

    data888['SOC_last'] = data888['SOC'].shift()
    data888['SOC_increase'] = data888.apply(lambda _: int(_['SOC'] > _['SOC_last']), axis=1)

    # data888['poschange'] = data888.apply(lambda _: _['new_poschange'] if not math.isnan(_['new_poschange']) else _['poschange'], axis=1)
    data888['poschange'] = data888.apply(lambda _: update(_['time'], _['poschange'], _['new_poschange'], _['time_end']),
                                         axis=1)
    data888['pcs'] = data888['poschange'].shift(-1)
    data888['abs'] = data888.apply(lambda _: abs(_['pcs'] - _['poschange']), axis=1).shift()
    data888['posid'] = data888.groupby(['car_id'])['abs'].cumsum()
    # data888['posid'][0] = 0
    data888.drop(['pcs', 'abs'], axis=1, inplace=True)


    def count_dis_by_tbdd(df):
        df['lon_last'] = df['lon'].shift()
        df['lat_last'] = df['lat'].shift()
        df.dropna(subset=['lon', 'lat', 'lon_last', 'lat_last'], inplace=True)
        return int(sum([tbd.getdistance(__['lon'], __['lat'], __['lon_last'], __['lat_last']) for __ in df.iloc]))


    def charge_start_and_end(df):
        _ = df.loc[df.SOC_increase == 1, 'time']
        return None if _.empty else (_.iat[0], _.iat[-1])


    gb = data888.groupby(['car_id', 'posid'])
    data22 = pd.concat([gb['time'].apply(lambda _: _.iloc[0]), gb['time'].apply(lambda _: _.iloc[-1]),
                        gb['SOC'].apply(lambda _: _.iloc[0]), gb['SOC'].apply(lambda _: _.iloc[-1]),
                        gb['lon'].apply(lambda _: _.iloc[0]), gb['lat'].apply(lambda _: _.iloc[0]),
                        gb['lon'].apply(lambda _: _.iloc[-1]), gb['lat'].apply(lambda _: _.iloc[-1]),
                        gb.apply(count_dis_by_tbdd), gb.apply(charge_start_and_end)], axis=1)
    data22.columns = ['time_start', 'time_end', 'SOC_start', 'SOC_end', 'lon_start', 'lat_start', 'lon_end', 'lat_end', 'dis_sum', 'charge_start_and_end']
    data22['charge_start_and_end'] = data22.apply(lambda _: _['charge_start_and_end'] if _['dis_sum'] == 0 else None, axis=1)
    data22.reset_index(inplace=True)

    data33 = data888.drop(columns=['time_end', 'new_poschange', 'time', 'SOC', 'lon', 'lat', 'SOC_increase'])
    data33 = data33.drop_duplicates(subset=['car_id', 'posid']).dropna()

    data44 = pd.merge(left=data22, right=data33)

    # 计算 time_duration
    def count_time_duration(_):
        total_sec = (_['time_end'] - _['time_start']).total_seconds()
        return total_sec // 3600, (total_sec % 3600) // 60, total_sec % 60

    data5 = data44.copy()
    data5['time_duration_h_m_s'] = data5.apply(count_time_duration, axis=1)

    run = data5[data5['dis_sum'] != 0]
    run.reset_index(inplace=True)
    del run['charged_state']
    del run['poschange']
    del run['index']
    run.to_csv('./data_csv/run/' + file + '.csv')
    tbd.dumpjson(run.to_json(orient="records"), './data_json/run/' + file + '.json')

    stop = data5[data5['dis_sum'] == 0]
    stop.reset_index(inplace=True)
    del stop['poschange']
    del stop['dis_sum']
    del stop['index']
    stop.to_csv('./data_csv/stop/' + file + '.csv')
    tbd.dumpjson(stop.to_json(orient="records"), './data_json/stop/' + file + '.json')


    def count_charge_duration(_):
        total_sec = (_['time_end_charge'] - _['time_start_charge']).total_seconds()
        return total_sec // 3600, (total_sec % 3600) // 60, total_sec % 60

    charged = stop[stop['SOC_start'] < stop['SOC_end']]
    charged[['time_start_charge', 'time_end_charge']] = charged['charge_start_and_end'].apply(pd.Series)
    del charged['charge_start_and_end']
    charged['charge_duration_h_m_s'] = charged.apply(count_charge_duration, axis=1)
    charged.to_csv('./data_csv/charged/' + file + '.csv')
    tbd.dumpjson(charged.to_json(orient="records"), './data_json/charged/' + file + '.json')

    stop_not_charged = stop[stop['SOC_start'] >= stop['SOC_end']]
    del stop_not_charged['charge_start_and_end']
    stop_over_1h_not_charged = stop_not_charged[stop_not_charged['time_duration_h_m_s'].apply(lambda _: _[0] > 0)]
    stop_over_1h___not_charged___SOC_less_than_40 = stop_over_1h_not_charged[stop_over_1h_not_charged['SOC_start'] < 40]
    stop_over_1h___not_charged___SOC_less_than_40.to_csv('./data_csv/potential/' + file + '.csv')
    tbd.dumpjson(stop_over_1h___not_charged___SOC_less_than_40.to_json(orient="records"), './data_json/potential/' + file + '.json')

    print(file)
