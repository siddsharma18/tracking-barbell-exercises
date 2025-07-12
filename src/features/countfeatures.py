import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  # Use sklearn's in-built normalizer
from scipy.signal import butter, lfilter, filtfilt
import pandas as pd

# This class removes the high-frequency data (considered noise) from the data.
class LowPassFilter:

    def low_pass_filter(data_table, col, sampling_frequency, cutoff_frequency, order=5, phase_shift=True):
        # Nyquist frequency is half the sampling frequency
        nyq = 0.5 * sampling_frequency
        cut = cutoff_frequency / nyq  # Normalize the frequency

        # Design a low-pass Butterworth filter
        b, a = butter(order, cut, btype='low', analog=False)
        
        # Apply zero-phase filtering to avoid phase shift or regular filtering if needed
        if phase_shift:
            data_table[col + '_lowpass'] = filtfilt(b, a, data_table[col])  # Zero-phase filter
        else:
            data_table[col + '_lowpass'] = lfilter(b, a, data_table[col])  # Regular filter
        return data_table

# Class for Principal Component Analysis (PCA)
# We can only apply PCA when there are no missing values (i.e., no NaN).
# Missing values must be imputed beforehand.
class PrincipalComponentAnalysis:

    def __init__(self):
        self.pca = PCA()

    # Normalize the data using StandardScaler (replaces util.normalize_dataset)
    def normalize_dataset(self, data_table, cols):
        scaler = StandardScaler()
        # Normalize the columns of interest
        data_table[cols] = scaler.fit_transform(data_table[cols])
        return data_table

    # Perform PCA on selected columns and return explained variance
    def determine_pc_explained_variance(self, data_table, cols):
        # Normalize the data before PCA
        dt_norm = self.normalize_dataset(data_table, cols)

        # Perform PCA
        self.pca = PCA(n_components=len(cols))
        self.pca.fit(dt_norm[cols])
        
        # Return the explained variances
        return self.pca.explained_variance_ratio_

    # Apply PCA and add new PCA columns to the data table
    def apply_pca(self, data_table, cols, number_comp):
        # Normalize the data
        dt_norm = self.normalize_dataset(data_table, cols)

        # Perform PCA
        self.pca = PCA(n_components=number_comp)
        self.pca.fit(dt_norm[cols])

        # Transform the old values into principal components
        new_values = self.pca.transform(dt_norm[cols])

        # Add the new PCA component columns to the data table
        for comp in range(number_comp):
            data_table['pca_' + str(comp + 1)] = new_values[:, comp]

        return data_table
    
    
pd.options.mode.chained_assignment=None

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"]=(20,5)
plt.rcParams["figure.dpi"]=100
plt.rcParams["lines.linewidth"]=2

dataframe= pd.read_pickle("/Users/siddharthsharma/Desktop/tracking-barbell-exercises/data/interim/01_data_processed.pkl")
dataframe=dataframe[dataframe["label"]!= "rest"]

acc_r=dataframe["acc_x"]**2 + dataframe["acc_y"]**2 + dataframe["acc_z"]**2
gyr_r=dataframe["gyr_x"]**2 + dataframe["gyr_y"]**2 + dataframe["gyr_z"]**2
dataframe["gyr_r"]=np.sqrt(gyr_r)
dataframe["acc_r"]=np.sqrt(acc_r)

dataframe_squat=dataframe[dataframe["label"]== "squat"]
dataframe_bench=dataframe[dataframe["label"]== "bench"]
dataframe_dead=dataframe[dataframe["label"]== "dead"]
dataframe_ohp=dataframe[dataframe["label"]== "ohp"]
dataframe_row=dataframe[dataframe["label"]== "row"]

plot_dataframe=dataframe_bench

plot_dataframe[plot_dataframe["set"]==plot_dataframe["set"].unique()[0]]["acc_x"].plot()
plot_dataframe[plot_dataframe["set"]==plot_dataframe["set"].unique()[0]]["acc_y"].plot()
plot_dataframe[plot_dataframe["set"]==plot_dataframe["set"].unique()[0]]["acc_z"].plot()
plot_dataframe[plot_dataframe["set"]==plot_dataframe["set"].unique()[0]]["acc_r"].plot()

plot_dataframe[plot_dataframe["set"]==plot_dataframe["set"].unique()[0]]["gyr_x"].plot()
plot_dataframe[plot_dataframe["set"]==plot_dataframe["set"].unique()[0]]["gyr_y"].plot()
plot_dataframe[plot_dataframe["set"]==plot_dataframe["set"].unique()[0]]["gyr_z"].plot()
plot_dataframe[plot_dataframe["set"]==plot_dataframe["set"].unique()[0]]["gyr_r"].plot()

plt.show()


fs=1000/200

LowPass=LowPassFilter

bench_set=dataframe_bench[dataframe_bench["set"]==dataframe_bench["set"].unique()[0]]
squat_set=dataframe_squat[dataframe_squat["set"]==dataframe_squat["set"].unique()[0]]
dead_set=dataframe_dead[dataframe_dead["set"]==dataframe_dead["set"].unique()[0]]
row_set=dataframe_row[dataframe_row["set"]==dataframe_row["set"].unique()[0]]
ohp_set=dataframe_ohp[dataframe_ohp["set"]==dataframe_ohp["set"].unique()[0]]

bench_set["acc_r"].plot()
plt.show()

column="acc_y"
LowPass.low_pass_filter(bench_set, col=column, sampling_frequency=fs, cutoff_frequency=0.4, order=10)[column + "_lowpass"].plot()

plt.show()


def count_reps(dataset, cutoff=0.4, order=10, column="acc_r"):

    data=LowPass.low_pass_filter(dataset, col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=order)
    indexes=argrelextrema(data[column + "_lowpass"].values, np.greater)
    peaks=data.iloc[indexes]
    
    fig, ax = plt.subplots()
    plt.plot(dataset[f"{column}_lowpass"])
    plt.plot(peaks[f"{column}_lowpass"], "o", color="red")
    ax.set_ylabel(f"{column}_lowpass")
    exercise = dataset["label"].iloc[0].title()
    category = dataset["category"].iloc[0].title()
    plt.title(f"{category} {exercise}: {len(peaks)} Reps")
    plt.show()
    
    return len(peaks)
    
count_reps(bench_set,cutoff=0.4)
count_reps(squat_set,cutoff=0.35)
count_reps(row_set,cutoff=0.65, column="gyr_x")
count_reps(ohp_set,cutoff=0.35)
count_reps(dead_set,cutoff=0.4)

dataframe["reps"]=dataframe["category"].apply(lambda x: 5 if x=="heavy" else 10)
rep_dataframe=dataframe.groupby(["label", "category", "set"])["reps"].max().reset_index()
rep_dataframe["reps_pred"]=0

for s in dataframe["set"].unique():
    subset=dataframe[dataframe["set"]==s]
    column="acc_r"
    cutoff=0.4
    
    if subset["label"].iloc[0]=="squat":
        cutoff=0.35
        
    if subset["label"].iloc[0]=="row":
        cutoff=0.65
        col="gyr_x"
        
    if subset["label"].iloc[0]=="ohp":
        cutoff=0.35
        
    reps=count_reps(subset, cutoff=cutoff, column=column)
    rep_dataframe.loc[rep_dataframe["set"]==s, "reps_pred"]=reps
    
rep_dataframe

error=mean_absolute_error(rep_dataframe["reps"], rep_dataframe["reps_pred"]).round(2)
rep_dataframe.groupby(["label", "category"])["reps", "reps_pred"].mean().plot.bar()