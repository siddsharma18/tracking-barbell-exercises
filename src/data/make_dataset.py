import pandas as pd
from glob import glob


singlefile_acceleration= pd.read_csv("/Users/siddharthsharma/Desktop/tracking-barbell-exercises/data/raw/MetaMotion /A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")

singlefile_gyroscope= pd.read_csv("/Users/siddharthsharma/Desktop/tracking-barbell-exercises/data/raw/MetaMotion /A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")

files= glob("/Users/siddharthsharma/Desktop/tracking-barbell-exercises/data/raw/MetaMotion /*.csv")
len(files)

datapath= "/Users/siddharthsharma/Desktop/tracking-barbell-exercises/data/raw/MetaMotion /"
f=files[0]

f.split("-")

participant=f.split("-")[2].lstrip("exercises/data/raw/MetaMotion /")
label = f.split("-")[3]
category= f.split("-")[4].rstrip("_MetaWear_2019").rstrip("123")


dataframe= pd.read_csv(f)

dataframe["participant"]= participant
dataframe["label"]=label
dataframe["category"]=category

accelerometer_dataframe = pd.DataFrame()
gyroscope_dataframe = pd.DataFrame()

acc_set=1
gyr_set=1

for f in files:
    participant=f.split("-")[2].lstrip("exercises/data/raw/MetaMotion /")
    label = f.split("-")[3]
    category= f.split("-")[4].rstrip("_MetaWear_2019").rstrip("123")

    
    dataframe= pd.read_csv(f)

    dataframe["participant"]= participant
    dataframe["label"]=label
    dataframe["category"]=category
    
    if "Accelerometer" in f:
        dataframe["set"]=acc_set
        acc_set+=1
        accelerometer_dataframe= pd.concat([accelerometer_dataframe, dataframe])
        
    if "Gyroscope" in f:
        dataframe["set"]=gyr_set
        gyr_set+=1
        gyroscope_dataframe= pd.concat([gyroscope_dataframe, dataframe])    
        
accelerometer_dataframe.info()

pd.to_datetime(dataframe["epoch (ms)"], unit="ms")

accelerometer_dataframe.index=pd.to_datetime(accelerometer_dataframe["epoch (ms)"], unit="ms")
gyroscope_dataframe.index=pd.to_datetime(gyroscope_dataframe["epoch (ms)"], unit="ms")

del accelerometer_dataframe["epoch (ms)"]
del accelerometer_dataframe["time (01:00)"]
del accelerometer_dataframe["elapsed (s)"]


del gyroscope_dataframe["epoch (ms)"]
del gyroscope_dataframe["time (01:00)"]
del gyroscope_dataframe["elapsed (s)"]


files= glob("/Users/siddharthsharma/Desktop/tracking-barbell-exercises/data/raw/MetaMotion /*.csv")

def read_data_from_files(files):
    
    accelerometer_dataframe = pd.DataFrame()
    gyroscope_dataframe = pd.DataFrame()

    acc_set=1
    gyr_set=1

    for f in files:
        participant=f.split("-")[2].lstrip("exercises/data/raw/MetaMotion /")
        label = f.split("-")[3]
        category= f.split("-")[4].rstrip("_MetaWear_2019").rstrip("123")

    
        dataframe= pd.read_csv(f)

        dataframe["participant"]= participant
        dataframe["label"]=label
        dataframe["category"]=category
    
        if "Accelerometer" in f:
            dataframe["set"]=acc_set
            acc_set+=1
            accelerometer_dataframe= pd.concat([accelerometer_dataframe, dataframe])
        
        if "Gyroscope" in f:
            dataframe["set"]=gyr_set
            gyr_set+=1
            gyroscope_dataframe= pd.concat([gyroscope_dataframe, dataframe])    
        

    accelerometer_dataframe.index=pd.to_datetime(accelerometer_dataframe["epoch (ms)"], unit="ms")
    gyroscope_dataframe.index=pd.to_datetime(gyroscope_dataframe["epoch (ms)"], unit="ms")

    del accelerometer_dataframe["epoch (ms)"]
    del accelerometer_dataframe["time (01:00)"]
    del accelerometer_dataframe["elapsed (s)"]


    del gyroscope_dataframe["epoch (ms)"]
    del gyroscope_dataframe["time (01:00)"]
    del gyroscope_dataframe["elapsed (s)"]

    return accelerometer_dataframe, gyroscope_dataframe


accelerometer_dataframe, gyroscope_dataframe=read_data_from_files(files)
data_merged=pd.concat([accelerometer_dataframe.iloc[:,:3], gyroscope_dataframe], axis=1)

data_merged.columns = [
"acc_x",
"acc_y",
"acc_z",
"gyr_x",
"gyr_y",
"gyr_z",
"participant",
"label",
"category",
"set",
]   

sampling={
"acc_x":"mean",
"acc_y":"mean",
"acc_z":"mean",
"gyr_x":"mean",
"gyr_y":"mean",
"gyr_z":"mean",
"participant":"last",
"label":"last",
"category":"last",
"set":"last"
}

data_merged[:1000].resample(rule="200ms").apply(sampling)

days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D" ))]
data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days] )
data_resampled ["set"] = data_resampled ["set"]. astype("int")
data_resampled.info()

data_resampled.to_pickle("/Users/siddharthsharma/Desktop/tracking-barbell-exercises/data/interim/02_data_processed.pkl")
