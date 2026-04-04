

# This file is for PAMAP2 data processing
import pandas as pd
import numpy as np
import h5py
import torch
from random import shuffle

activities = {
    1: "stand",
    2: "walk",
    4: "sit",
    5: "lie"
}


def read_files(current_path_in_repo, path_to_opportunity_folder):
    """
    read_files
        .dat file -> csv with whitespaces instead of commas
    """
    # pick partial data from dataset
    list_of_files = [
        "./dataset/S1-ADL1.dat",
        "./dataset/S1-ADL2.dat",
        "./dataset/S1-ADL3.dat",
        "./dataset/S1-ADL4.dat",
        "./dataset/S1-ADL5.dat",
        "./dataset/S1-Drill.dat",

        "./dataset/S2-ADL1.dat",
        "./dataset/S2-ADL2.dat",
        "./dataset/S2-ADL3.dat",
        "./dataset/S2-ADL4.dat",
        "./dataset/S2-ADL5.dat",
        "./dataset/S2-Drill.dat",

        "./dataset/S3-ADL1.dat",
        "./dataset/S3-ADL2.dat",
        "./dataset/S3-ADL3.dat",
        "./dataset/S3-ADL4.dat",
        "./dataset/S3-ADL5.dat",
        "./dataset/S3-Drill.dat",

        "./dataset/S4-ADL1.dat",
        "./dataset/S4-ADL2.dat",
        "./dataset/S4-ADL3.dat",
        "./dataset/S4-ADL4.dat",
        "./dataset/S4-ADL5.dat",
        "./dataset/S4-Drill.dat",

    ]


    list_of_drill = [
        "./dataset/S1-Drill.dat",
        "./dataset/S2-Drill.dat",
        "./dataset/S3-Drill.dat",
        "./dataset/S4-Drill.dat",
    ]

    list_of_files = list(
        map(
            lambda path_to_file: path_to_opportunity_folder + path_to_file[1:],
            list_of_files,
        )
    )
    list_of_drill = list(
        map(
            lambda path_to_file: path_to_opportunity_folder + path_to_file[1:],
            list_of_drill,
        )
    )

    col_names = []
    with open(current_path_in_repo + "/column_names.txt", "r") as file:
        lines = file.read().splitlines()
        for line in lines:

            col_names.append(line.split(' ',2)[2:][0])
    data_collection = pd.DataFrame()
    for i, file in enumerate(list_of_files):
        print(file, " is reading...")
        proc_data = pd.read_table(file, header=None, sep="\s+")
        proc_data.columns = col_names
        proc_data["file_index"] = i  # put the file index at the end of the row
        data_collection = data_collection.append(proc_data, ignore_index=True)
        # break; # for testing short version, need to delete later
    data_collection.reset_index(drop=True, inplace=True)
    # print(data_collection)
    return data_collection


def data_cleaning(dataCollection):
    """
    data cleaning
    """
    # print(dataCollection.shape)
    # dataCollection = dataCollection.loc[:, dataCollection.isnull().mean() < 0.1]  # drop the columns which has NaN over 10%
    idx1 = range(1, 34)
    idx2 = range(37, 102)
    idx3 = range(231, 244)
    idx = list(idx1) + list(idx2)+list(idx3)+[-1]

    dataCollection = dataCollection.iloc[:, idx]  # drop the columns which has NaN over 10%
    # print('df', dataCollection.shape)
    # dataCollection = dataCollection.loc[:, dataCollection.isnull().mean()<1e-8]
    # print(list(dataCollection.columns.values))
    # print(dataCollection.shape)

    dataCollection = dataCollection.apply(
        pd.to_numeric, errors="coerce"
    )  # removal of non numeric data in cells

    print(
        "number of NaN before interpolation", dataCollection.isna().sum().sum()
    )  # count all NaN

    dataCollection = dataCollection.interpolate(method ='linear')
    dataCollection = dataCollection.fillna(method='backfill')


    """
    before:
            a	b	 c	  d
        0	1	4.0	 8.0  NaN
        1	2	NaN	 NaN  9.0
        2	3	6.0	 NaN  NaN
        3	4	6.0	 7.0  8.0

    after:
            a	b	c	        d
        0	1	4.0	8.000000	NaN
        1	2	5.0	7.666667	9.0
        2	3	6.0	7.333333	8.5
        3	4	6.0	7.000000	8.0

    -> standard linear interpolation
    -> we use fillna (taking the last availble value, duplicate it)
    -> what interpolation for what sensor makes semantic sense? quaternion? acceleration?

    """
    print(
        "number of NaN after interpolation", dataCollection.isna().sum().sum()
    )  # count all NaN
    # removal of any remaining NaN value cells by constructing new data points in known set of data points

    print("data cleaned!")
    return dataCollection


def reset_label(dataCollection, locomotion):
    # Convert original labels {1, 2, 4, 5, 101, 102, 103, 104, 105} to new labels.

    # CHANGE??!! before the NULL Acitivity was filtered out!!1
    without_null_mapping = { 
        1: 1,
        2: 2,
        5: 0,
        4: 3

    }  # old activity id to new activity Id

    with_null_mapping = {
        0: 0,
        101: 1,
        102: 2,
        103: 3,
        104: 4,
        105: 5,
    }  # old activity id to new activity Id

    if locomotion:  # new labels [0,1,2,3]
        for i in [5, 4]:  # reset ids in Locomotion column
            dataCollection.loc[dataCollection.Locomotion == i, "Locomotion"] = without_null_mapping[
                i
            ]
    else:  # reset the high level activities ; new labels [0,1,2,3,4]
        for j in [101, 102, 103, 104, 105]:  # reset ids in HL_activity column
            dataCollection.loc[
                dataCollection.HL_Activity == j, "HL_Activity"
            ] = with_null_mapping[j]
    return dataCollection


def segment_locomotion(
    dataCollection, window_size
):  # segment the data and create a dataset with locomotion classes as labels
    # remove locomotions with 0
    def segment_ind(data):
        n = len(data)
        X = []
        y = []
        start = 0
        end = 0
        while start + window_size - 1 < n:
            end = start + window_size - 1
            if (
                    data[start][loco_i] == data[end][loco_i]
                    and data[start][-1] == data[end][-1]
            ):  # if the frame contains the same activity and from the file
                X.append(data[start: (end + 1), 0:loco_i])
                y.append(data[start][loco_i])
                start += int(window_size // 2)  # 50% overlap
            else:  # if the frame contains different activities or from different objects, find the next start point
                while start + window_size - 1 < n:
                    if data[start][loco_i] != data[start + 1][loco_i]:
                        break
                    start += 1
                start += 1

        num,_,num_channels = np.shape(X)
        X_ = np.reshape(X, [-1, num_channels])
        X_mean = np.mean(X_, 0)
        X_std = np.std(X_, 0)
        print(X_mean)
        print(X_std)
        X = (X_-X_mean)/X_std
        X = np.reshape(X, [num, window_size, num_channels])


        num = len(X)
        idx = np.arange(num)
        np.random.shuffle(idx)
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)
        X = X[idx]
        y = y[idx]

        return X[:(num-200)], y[:(num-200)], X[(num-200):], y[(num-200):]


    dataCollection = dataCollection.drop(
        dataCollection[dataCollection.Locomotion == 0].index
    )
    # reset labels
    dataCollection = reset_label(dataCollection, True)
    # print(dataCollection.columns)
    loco_i = dataCollection.columns.get_loc("Locomotion")
    # convert the data frame to numpy array
    # print(dataCollection)
    # print(dataCollection.shape)
    # print(
    #     "number of NaN after interpolation", dataCollection.isna().sum().sum()
    # )  # count all NaN
    # dataCollection = dataCollection.interpolate(method='linear',limit_direction='forward')
    # print(
    #     "number of NaN after interpolation", dataCollection.isna().sum().sum()
    # )  # count all NaN
    data = dataCollection.to_numpy()
    # print(np.shape(data))
    # print(data[0])
    # segment the data
    s_1 = []
    s_2 = []
    s_3 = []
    s_4 = []
    # print(data[:][-1])
    n = len(data)
    num_file = 6
    for i in range(n):
        idx = data[i][-1]
        # print()
        if idx<num_file:
            s_1.append(data[i])
        elif idx>=num_file and idx<num_file*2:
            s_2.append(data[i])
        elif idx>=num_file*2 and idx<num_file*3:
            s_3.append(data[i])
        elif idx>=num_file*3 and idx<num_file*4:
            s_4.append(data[i])

    s_1 = np.stack(s_1,0)
    s_2 = np.stack(s_2,0)
    s_3 = np.stack(s_3,0)
    s_4 = np.stack(s_4,0)
    # print(np.shape(s_1)[0])
    # print(np.shape(s_2)[0])
    # print(np.shape(s_3)[0])
    # print(np.shape(s_4)[0])

    train_X_1,train_y_1,test_X_1,test_y_1 = segment_ind(s_1)
    train_X_2,train_y_2,test_X_2,test_y_2 = segment_ind(s_2)
    train_X_3,train_y_3,test_X_3,test_y_3 = segment_ind(s_3)
    train_X_4,train_y_4,test_X_4,test_y_4 = segment_ind(s_4)


    return {"samples": np.asarray(train_X_1), "labels": np.asarray(train_y_1, dtype=int)}, \
           {"samples": np.asarray(test_X_1), "labels": np.asarray(test_y_1, dtype=int)}, \
           {"samples": np.asarray(train_X_2), "labels": np.asarray(train_y_2, dtype=int)}, \
           {"samples": np.asarray(test_X_2), "labels": np.asarray(test_y_2, dtype=int)}, \
           {"samples": np.asarray(train_X_3), "labels": np.asarray(train_y_3, dtype=int)}, \
           {"samples": np.asarray(test_X_3), "labels": np.asarray(test_y_3, dtype=int)}, \
           {"samples": np.asarray(train_X_4), "labels": np.asarray(train_y_4, dtype=int)}, \
            {"samples": np.asarray(test_X_4), "labels": np.asarray(test_y_4, dtype=int)}

def segment_high_level(
    dataCollection, window_size
):  # segment the data and create a dataset with high level activities as labels
    # remove HL_activities with 0
    # remember to change the mapping in reset_label
    # dataCollection = dataCollection.drop(
    #     dataCollection[dataCollection.HL_Activity == 0].index # THIS REMOVES NULL ACTIVITIES!!!!!!!!!!!
    # )
    # reset labels
    dataCollection = reset_label(dataCollection, False)
    """
    relabeling
        old labels: 
            locomotion: {0, 1, 2, 4, 5} 
            high level: {0, 101, 102, 103, 104, 105}
        new labels:
            mapping: {1:1, 2:2, 5:0, 4:3, 101: 0, 102:1, 103:2, 104:3, 105:4}
    """
    # print(dataCollection.columns)
    HL_Activity_i = dataCollection.columns.get_loc("HL_Activity")
    # convert the data frame to numpy array
    data = dataCollection.to_numpy()
    # segment the data
    n = len(data)
    X = []
    y = []
    start = 0
    end = 0
    while start + window_size - 1 < n:
        end = start + window_size - 1

        # has planned window the same activity in the beginning and the end, is from the same file in the beginning and the end
        # what if it changes back and forth?
        if (
            data[start][HL_Activity_i] == data[end][HL_Activity_i]
            and data[start][-1] == data[end][-1] # the last index is the file index
        ):

            # print(data[start:(end+1),0:(HL_Activity_i)])
            # first part time axis, second part sensor axis -> get window
            X.append(
                data[start : (end + 1), 0 : (HL_Activity_i - 1)] # data[timeaxis/row, featureaxis/column] data[1, 2] gives specific value, a:b gives you an interval
            )  # slice before locomotion
            y.append(data[start][HL_Activity_i])  # the first data point is enough
            start += window_size // 2  # 50% overlap!!!!!!!!!

        # if the frame contains different activities or from different objects, find the next start point
        # if there is a rest smaller than the window size -> skip (window small enough?)
        else:
            while start + window_size - 1 < n:
                # find the switch point -> the next start point
                # different file check missing! will come here again (little overhead)
                if data[start][HL_Activity_i] != data[start + 1][HL_Activity_i]:
                    break
                start += 1
            start += 1  # dirty fix for the missing 'different file' check?
    print(np.asarray(X).shape, np.asarray(y).shape)
    return {"inputs": np.asarray(X), "labels": np.asarray(y, dtype=int)}


def plot_series(df, colname, act, file_index, start, end):
    unit = "ms^-2"
    # pylim =(-25,25)
    # print(df.head())
    print(set(df.loc[df.file_index == file_index, "Locomotion"]))
    df1 = df[(df.Locomotion == act) & (df.file_index == file_index)]
    # df1 = df[(df.HL_Activity ==act) & (df.file_index == file_index)]
    if df1.shape[0] < 1:
        print("Didn't find the region. Please reset activityID and subject_id")
        return
    df_len = df1.shape[0]
    if df_len > start and df_len > end:
        df1 = df1[start:end]
    elif df_len > start and df_len <= end:
        df1 = df1[start:df_len]
    else:
        print("Out of boundary, please reset the start and end points")
        return
    print(df1.shape)
    # print(df1.head(10))
    plottitle = colname + " - " + str(act)
    # plotx = colname
    fig = df1[colname].plot()
    # print(df.index)
    # ax1 = df1.plot(x=df.index,y=plotx, color='r', figsize=(12,5), ylim=pylim)
    fig.set_title(plottitle)
    fig.set_xlabel("window")
    fig.set_ylabel(unit)


def save_data(data, file_name, current_path_in_repo):  # save the data in h5 format
    f = h5py.File(current_path_in_repo + "/" + file_name, "w")
    for key in data:
        print(key)
        f.create_dataset(key, data=data[key])
    f.close()
    print("Done.")

def save_data_pt(data, file_name, current_path_in_repo):  # save the data in h5 format

    # torch.save(data[:(num-200)], current_path_in_repo + "/" + 'train_'+file_name +'.pt')
    torch.save(data, current_path_in_repo + "/" +file_name +'.pt')

    print("Saving dataset {} is Done.".format((file_name)))
if __name__ == "__main__":
    window_size = 128

    path_to_opportunity_folder = "raw_datasets/OpportunityUCIDataset"
    current_path_in_repo = "datasets/OpportunityUCIDataset"

    df = read_files(current_path_in_repo, path_to_opportunity_folder)

    # print('df shape is',df.shape)
    df = data_cleaning(df)  # drop columns, interpolate NaN
    # print('df shape is',df.shape)


    # loco_filename = "loco_2.h5"  # "loco.h5" is to save locomotion dataset.
    train_s_1,test_s_1, train_s_2,test_s_2, train_s_3,test_s_3, train_s_4,test_s_4 = segment_locomotion(df, window_size)
    save_data_pt(train_s_1, 'train_subject_1', current_path_in_repo)
    save_data_pt(test_s_1, 'test_subject_1', current_path_in_repo)
    save_data_pt(train_s_2, 'train_subject_2', current_path_in_repo)
    save_data_pt(test_s_2, 'test_subject_2', current_path_in_repo)
    save_data_pt(train_s_3, 'train_subject_3', current_path_in_repo)
    save_data_pt(test_s_3, 'test_subject_3', current_path_in_repo)
    save_data_pt(train_s_4, 'train_subject_4', current_path_in_repo)
    save_data_pt(test_s_4, 'test_subject_4', current_path_in_repo)


    # hl_filename = "hl_2.h5"  # "hl.h5" is to save high level dataset
    # data_hl = segment_high_level(df, window_size)
    """
        without subject 5 and subset of sensors:
            data_hl['inputs'].shape # (34181, 25, 220) -> 34181 windows, 25 timestamps, 220 features (sensors)
            data_hl['labels'].shape # (34181,) -> 34181 labels

        with subject 5 and subset of sensors:
            data_hl['inputs'].shape # (49484, 25, 51)
            data_hl['labels'].shape # (49484,)
    """
    # save_data(data_hl, hl_filename, current_path_in_repo)
