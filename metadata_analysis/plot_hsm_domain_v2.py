# built-in
import datetime
import json
import os

# external
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, dates, gridspec

# local
from eval_quality import get_quality_df

main_dir = 'quality'

def patient_function(patients):

    pat_dir = main_dir + os.sep + patients + os.sep + 'biosignals'

    files = os.listdir(pat_dir)
    file1 = Biosignal.load(pat_dir + os.sep + files[-1])

    # event1 = Event('crise 1', datetime.datetime(2021, 3, 30, 22, 39))

    # file1.associate(event1)

    # crop_file = file1[datetime.timedelta(minutes=2):'crise 1':datetime.timedelta(minutes=2)]

    # filter1 = TimeDomainFilter(ConvolutionOperation.HAMMING, window_length=datetime.timedelta(seconds=5))
    # crop_file._accept_filtering(filter1)

    table_long = pd.DataFrame(columns=['Time'])

    for segment in file1[:].segments:
        aa_time = pd.date_range(segment.initial_datetime, segment.final_datetime, freq='1s')
        table_long = pd.concat((table_long, pd.DataFrame({'Time': aa_time})), ignore_index=True)

    table_long.to_csv('C:\\Users\\Mariana\\Documents\\CAT\\data_domains' + os.sep + 'domain_' + patients + '.csv')
    print('je')


def get_timeline(df):
    """
    Transform a dataframe with timestamps into a timeline table
    :param signal_check: dataframe with timestamps

    """
    timeline_table = pd.DataFrame(columns=['start', 'end', 'quality'])
    time_jumps = np.hstack([0, np.argwhere(np.diff(df['times_utc']).astype(int)>10000000000)[:,0], len(df)])
    start_date = df['times_utc'].iloc[0]
    for i in range(len(time_jumps)-1):
        start = start_date + datetime.timedelta(milliseconds=int(1000*(time_jumps[i] +1)/0.1))
        end = start_date + datetime.timedelta(milliseconds=int(1000*(time_jumps[i+1])/0.1))
        timeline_table.loc[i] = [start, end, 1]
    return timeline_table

week = ['S', 'M', 'T', 'W', 'T', 'F', 'S']

colors = ['#A9D1ED', '#843C0C', '#F8CBAD', '#B0B0B0']
fictional_date = datetime.datetime(2000, 1, 3, 9, 0, 0)
fig, ax1 = plt.subplots(figsize=(15, 10))

patient_dict = json.load(open('patient_info.json'))

patients = [filename.split('_')[0] for filename in os.listdir('quality') if ('wearable' in filename and 'ecg' in filename)]
patients = [pat for pat in patients if patient_dict[pat]['source'] == 'HSM']

print(len(patients))

patients_year = {}

dir_ = 'C:\\Users\\Mariana\\Documents\\CAT\\data_domains\\HSM\\'
final_patients = {}

ytics_dict = {'PGSE': ['PC + ArmBit ForearmBit', '2017-04-23 21:50:21.105000'],
              'WMWV': ['PC + ArmBit ForearmBit', '2017-04-06 17:25:15.775000'],
              'QDST': ["PC + ArmBit ForearmBit", '2017-02-23 21:50:00.774000'],
              'OXDN': ['PC + ArmBit ForearmBit', '2019-02-12 21:44:32.796000'],
              #'Patient112': ['PC + ChestBit WristBit', '2020-03-03 18:57:57.772000'],
              'QFRK': ['PC + ChestBit WristBit', '2019-07-04 10:04:10.321000'],
              'LDXH': ['PC + ChestBit WristBit', '2019-05-07 10:20:10.321000'],
              'WOSQ': ['PC + ChestBit WristBit', '2019-02-26 11:10:35.275000'],
              'OQQA': ['PC + ChestBit WristBit', '2019-12-10 10:20:10.321000'],
              'UIJU': ['PC + ChestBit WristBit', '2019-07-30 10:20:10.321000'],
              #'Patient111': ['PC + ChestBit WristBit', '2020-02-18 12:02:39.799000'],
              'VNVW': ['PC + ChestBit WristBit', '2019-04-09 10:20:10.321000'],
              'RMJL': ['PC + ChestBit WristBit', '2019-08-27 16:17:18.854000'],
              'YWJN': ['PC + ChestBit WristBit', '2020-01-07 17:58:35.450000'],
              'RAFI': ['PC + ChestBit WristBit', '2019-03-12 10:58:29.403000'],
              'CNSV': ['EpiBOX + ChestBit', '2021-03-01 14:36:05.022688'],
              'MQWA': ['EpiBOX + ChestBit', '2021-09-22 16:21:14.498261'],
              'UWEF': ['EpiBOX + ChestBit', '2021-05-04 15:55:46.148100'],
              #'OQXF': ['EpiBOX + ChestBit', '2021-11-09 17:15:50.462852'],
              'AGGA': ['EpiBOX + ChestBit', '2021-12-13 13:39:30.879850'],
              #'RLJW': ['EpiBOX + ChestBit', '2021-11-23 16:00:00.000000'],
              'UDZG': ['EpiBOX + ChestBit', '2022-07-19 09:23:17.661779'],
              'BLIW': ['EpiBOX + ChestBit', '2022-10-24 15:54:56.728592'],
              'PRBQ': ['EpiBOX + ChestBit', '2022-07-04 15:29:19.722616'],
              'YIVL': ['EpiBOX + ChestBit', '2022-08-08 15:54:56.728592'],
              'MHDG': ['EpiBOX + ChestBit', '2022-05-23 15:54:56.728592'],
              'BSEA': ['EpiBOX + ChestBit', '2023-01-19 15:54:56.728592'],
              'GPPF': ['EpiBOX + ChestBit', '2022-12-12 15:54:56.728592'],
              'OFUF': ['EpiBOX + ChestBit', '2022-06-20 15:54:56.728592'],
              'RGNI': ['EpiBOX + ChestBit', '2022-11-21 15:54:56.728592'],              
}

final_patients, final_pos = [], []
patient_list = [pat for pat in patient_dict.keys() if (patient_dict[pat]['source'] == 'HSM' and patient_dict[pat]['wearable_dir'] != '')]

for pp, patient in enumerate(list(ytics_dict.keys())):
    print(patient)
    # dirhsm = os.path.join(dir_, patient + '_hospital_domain.parquet')
    # get hsm domain
    # if os.path.isfile(dirhsm):
    #    table_hsm = pd.read_parquet(dirhsm, engine='fastparquet')
    #else:
    #    table_hsm = None
    # get bit domain
    table_w = get_quality_df(patient, 'wearable', 'ecg')
    # get annotations
    try:
        excel_name = patient_dict[patient]['touch']
        excel_patient = pd.read_excel('/Volumes/T7 Touch/PreEpiSeizures/Patients_HSM/Patients_HSM_.xlsx', sheet_name=excel_name)
        excel_patient = excel_patient.loc[excel_patient['Crises'].notna()]
    except Exception as e:
        print(e)
        excel_patient = []

    if len(excel_patient) > 0:
        seizures = excel_patient.loc[excel_patient['Focal / Generalisada'].isin(['Focal', 'FWIA', 'F', 'FBTC', 
                                                                                 'FAS', 'FUAS', 'FIAS', 'FAS', 
                                                                                 'F/FWIA', 'Focal cognitiva ', 'Funk'])]
        subclinical = excel_patient.loc[excel_patient['Focal / Generalisada'].isin(['F(ME)', 'E'])]
        seizure_dates = pd.to_datetime(seizures['Data'], dayfirst=True)
        seizures_onset = [datetime.datetime.combine(seizure_dates.iloc[i], seizures['Hora Clínica'].iloc[i])
                          for i in range(len(seizure_dates))]
        subclinical_dates = pd.to_datetime(subclinical['Data'], dayfirst=True)
        subclinical_onset = [datetime.datetime.combine(subclinical_dates.iloc[i], subclinical['Hora Clínica'].iloc[i])
                          for i in range(len(subclinical_dates))]

    # table_dict = {'HSM': table_hsm, 'Bit': table_bit}
    initial_time = table_w['datetime'].iloc[0]
    last_time = table_w['datetime'].iloc[-1]

    times = table_w['datetime']
    delta = times.iloc[0] - initial_time
    # times = pd.to_datetime(times + abs(delta))

        # if times.iloc[0].year == 2009:
        #     times = times.loc[times > datetime.datetime(2010, 1, 1, 1, 1, 1)]
        #     if len(times) <= 0:
        #         times = pd.to_datetime(table['Time'])
        #     print('ok')
    initial_weekday = initial_time.weekday()
    if initial_weekday > 3:
        times -= datetime.timedelta(days=3)
        initial_datetime = times.iloc[0]
        initial_weekday = initial_time.weekday()
    week_axis = pd.date_range(initial_time.date() - datetime.timedelta(days=initial_weekday) + datetime.timedelta(hours=12), periods=6, freq='d')
    # week_axis = pd.date_range(initial_time.date(), last_time.date(), freq='d')

    week_utc = pd.to_datetime((week_axis-week_axis[0]).astype(np.int64))
    times_utc = pd.to_datetime((times - week_axis[0]).astype(np.int64))
    times_utc = times_utc.loc[times_utc <= week_utc[-1]]
    table_w['times_utc'] = times_utc
    df_w = get_timeline(table_w)

    #times_missing = pd.date_range(times_utc.iloc[0], times_utc.iloc[-1], freq='10s')
    #times_missing_DF_temp = pd.DataFrame({'Times missing': times_missing})
    #times_missing_DF = times_missing_DF_temp.loc[~times_missing_DF_temp['Times missing'].isin(pd.to_datetime(np.array(times_utc).astype('datetime64[s]')))]
    #missing_utc = times_missing_DF['Times missing']

    if len(excel_patient) > 0:
        seizures_utc = pd.to_datetime((pd.to_datetime(seizures_onset) - week_axis[0]).astype(np.int64))
        seizures_utc = [seizure for seizure in seizures_utc if seizure <= week_utc[-1]]
        subclinical_utc = pd.to_datetime((pd.to_datetime(subclinical_onset) - week_axis[0]).astype(np.int64))
        subclinical_utc = [subclinical for subclinical in subclinical_utc if subclinical <= week_utc[-1]]
        seizures_utc_times = np.sum([1 for i in range(len(seizures_utc))
                                        if len(times_utc[times_utc.between(seizures_utc[i], 
                                                                           seizures_utc[i] + datetime.timedelta(seconds=120))]) > 0])
        subclinical_utc_times = np.sum([1 for i in range(len(subclinical_utc)) 
                                        if len(times_utc[times_utc.between(subclinical_utc[i], 
                                                                           subclinical_utc[i] + datetime.timedelta(seconds=120))]) > 0])
        if len(seizures_utc) == 0:
            print('problem')
        print(f'{patient} captured seizures {seizures_utc_times} captured subclinical {subclinical_utc_times} '
                f'total seizures {len(seizures_utc)} total subclinical {len(subclinical_utc)}')

    ax1.set_xticks(week_utc, week[1:])
    missing_utc = pd.date_range(times_utc.iloc[0], times_utc.iloc[-1], freq='10s')
    week_utc_ = pd.date_range(week_utc[0], week_utc[-1], freq='10s')
    ax1.scatter(week_utc_, pp * np.ones(len(week_utc_)), linewidth=0.5, marker='_', c='lightgrey')
    ax1.scatter(missing_utc, pp * np.ones(len(missing_utc)), linewidth=1, marker='_', c=colors[2],
                    label='Missing', linestyle=(0, (10, 4)))
    ax1.scatter(times_utc, pp * np.ones(len(times_utc)), linewidth=5, marker='_', c=colors[0], label='Bitalino')

    #elif key == 'HSM':
    #    ax1.scatter(times_utc, pp * np.ones(len(times_utc)) - 0.05, s=10, marker='_', c=colors[2], label='Hospital')

    if len(excel_patient) > 0:
        ax1.scatter(seizures_utc, pp * np.ones(len(seizures_utc)) + 0.05, marker='*', c=colors[1], label='Seizures')
        ax1.scatter(subclinical_utc, pp * np.ones(len(subclinical_utc)) + 0.05, marker='*', c=colors[3], label='Subclinical')
    final_patients += [ytics_dict[patient][0]]
    final_pos += [pp]
    if len(final_patients) == 1:
        ax1.legend(loc='lower left')

# ticks should be arm and chest and wrist
ax1.set_yticks(final_pos, final_patients)
# fig.savefig('hsm_data_acquisition_domain_v2.png')
fig.suptitle('HSM Acquisition Domain of Wearable Data Per Patient', y=0.9)
fig.savefig('hsm_data_acquisition_domain.jpg', dpi=300, bbox_inches='tight')
