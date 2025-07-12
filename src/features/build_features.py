import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dataframe= pd.read_pickle("/Users/siddharthsharma/Desktop/tracking-barbell-exercises/data/interim/02_outliers_removed_chauvenets.pkl")

predictor_columns=list(dataframe.columns[:6])

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20,5)
plt.rcParams["figure.dpi"]=100
plt.rcParams["lines.linewidth"]=2

dataframe.info()

for col in predictor_columns:
    dataframe[col]=dataframe[col].interpolate()
    
dataframe.info()

dataframe[dataframe["set"]==25]["acc_y"]
dataframe[dataframe["set"]==50]["acc_y"]

duration=dataframe[dataframe["set"]==1].index[-1]-dataframe[dataframe["set"]==1].index[0]
duration.seconds

for s in dataframe["set"].unique():
    start=dataframe[dataframe["set"]==s].index[0]
    stop=dataframe[dataframe["set"]==s].index[-1]
    duration=stop-start
    dataframe.loc[(dataframe["set"]==s), "duration"]=duration.seconds
    
duration_dataframe=dataframe.groupby(["category"])["duration"].mean()
duration_dataframe.iloc[0]/5
duration_dataframe.iloc[1]/10

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  # Use sklearn's in-built normalizer
from scipy.signal import butter, lfilter, filtfilt
import pandas as pd

# This class removes the high-frequency data (considered noise) from the data.
class LowPassFilter:

    def low_pass_filter(self, data_table, col, sampling_frequency, cutoff_frequency, order=5, phase_shift=True):
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
    
##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

import numpy as np
import scipy.stats as stats

# Class to abstract a history of numerical values we can use as an attribute.
class NumericalAbstraction:

    # For the slope we need a bit more work.
    # We create time points, assuming discrete time steps with fixed delta t:
    def get_slope(self, data):
        
        times = np.array(range(0, len(data.index)))
        data = data.astype(np.float32)

        # Check for NaN's
        mask = ~np.isnan(data)

        # If we have no data but NaN we return NaN.
        if (len(data[mask]) == 0):
            return np.nan
        # Otherwise we return the slope.
        else:
            slope, _, _, _, _ = stats.linregress(times[mask], data[mask])
            return slope

    #TODO Add your own aggregation function here:
    # def my_aggregation_function(self, data) 

    # This function aggregates a list of values using the specified aggregation
    # function (which can be 'mean', 'max', 'min', 'median', 'std', 'slope')
    def aggregate_value(self, data, window_size, aggregation_function):
    # Compute the values and return the result based on the specified aggregation function.
        if aggregation_function == 'mean':
            return data.rolling(window=window_size, min_periods=window_size).mean()
        elif aggregation_function == 'max':
            return data.rolling(window=window_size, min_periods=window_size).max()
        elif aggregation_function == 'min':
            return data.rolling(window=window_size, min_periods=window_size).min()
        elif aggregation_function == 'median':
            return data.rolling(window=window_size, min_periods=window_size).median()
        elif aggregation_function == 'std':
            return data.rolling(window=window_size, min_periods=window_size).std()
        elif aggregation_function == 'slope':
            return data.rolling(window=window_size, min_periods=window_size).apply(self.get_slope)

        # Handle unsupported aggregation functions.
        else:
            return np.nan


    def abstract_numerical(self, data_table, cols, window_size, aggregation_function_name):
        for col in cols:
            # Applying the rolling window and then calling the aggregation function
            aggregations = self.aggregate_value(data_table[col], window_size, aggregation_function_name)
            # Store the aggregated values in a new column
            data_table[col + '_temp_' + aggregation_function_name + '_ws_' + str(window_size)] = aggregations
        
        return data_table

# Class to perform categorical abstraction. We obtain patterns of categorical attributes that occur frequently
# over time.
class CategoricalAbstraction:

    pattern_prefix = 'temp_pattern_'
    before = '(b)'
    co_occurs = '(c)'
    cache = {}

    # Determine the time points a pattern occurs in the dataset given a windows size.
    def determine_pattern_times(self, data_table, pattern, window_size):
        times = []

        # If we have a pattern of length one
        if len(pattern) == 1:
            # If it is in the cache, we get the times from the cache.
            if self.to_string(pattern) in self.cache:
                times = self.cache[self.to_string(pattern)]
            # Otherwise we identify the time points at which we observe the value.
            else:
               
                timestamp_rows = data_table[data_table[pattern[0]] > 0].index.values.tolist()
               
                times = [data_table.index.get_loc(i) for i in timestamp_rows]
                self.cache[self.to_string(pattern)] = times

        # If we have a complex pattern (<n> (b) <m> or <n> (c) <m>)
        elif len(pattern) == 3:
            # We computer the time points of <n> and <m>
            time_points_first_part = self.determine_pattern_times(data_table, pattern[0], window_size)
            time_points_second_part = self.determine_pattern_times(data_table, pattern[2], window_size)

            # If it co-occurs we take the intersection.
            if pattern[1] == self.co_occurs:
                # No use for co-occurences of the same patterns...
                if pattern[0] == pattern[2]:
                    times = []
                else:
                    times = list(set(time_points_first_part) & set(time_points_second_part))
            # Otherwise we take all time points from <m> at which we observed <n> within the given
            # window size.
            elif pattern[1] == self.before:
                for t in time_points_second_part:
                    if len([i for i in time_points_first_part if ((i >= t - window_size) & (i < t))]):
                        times.append(t)
        return times

    # Create a string representation of a pattern.
    def to_string(self, pattern):
        # If we just have one component, return the string.
        if len(pattern) == 1:
            return str(pattern[0])
        # Otherwise, return the merger of the strings of all
        # components.
        else:
            name = ''
            for p in pattern:
                name = name + self.to_string(p)
            return name

    # Selects the patterns from 'patterns' that meet the minimum support in the dataset
    # given the window size.
    def select_k_patterns(self, data_table, patterns, min_support, window_size):
        selected_patterns = []
        for pattern in patterns:
            # Determine the times at which the pattern occurs.
            times = self.determine_pattern_times(data_table, pattern, window_size)
            # Compute the support
            support = float(len(times))/len(data_table.index)
            # If we meet the minimum support, append the selected patterns and set the
            # value to 1 at which it occurs.
            if support >= min_support:
                selected_patterns.append(pattern)
                print(self.to_string(pattern))
                # Set the occurrence of the pattern in the row to 0.
                data_table[self.pattern_prefix + self.to_string(pattern)] = 0
                #data_table[self.pattern_prefix + self.to_string(pattern)][times] = 1
                data_table.iloc[times, data_table.columns.get_loc(self.pattern_prefix + self.to_string(pattern))] = 1
        return data_table, selected_patterns


    # extends a set of k-patterns with the 1-patterns that have sufficient support.
    def extend_k_patterns(self, k_patterns, one_patterns):
        new_patterns = []
        for k_p in k_patterns:
            for one_p in one_patterns:
                # Add a before relationship
                new_patterns.append([k_p, self.before, one_p])
                # Add a co-occurs relationship.
                new_patterns.append([k_p, self.co_occurs, one_p])
        return new_patterns


    # Function to abstract our categorical data. Note that we assume a list of binary columns representing
    # the different categories. We set whether the column names should match exactly 'exact' or should include the
    # specified name 'like'. We also express a minimum support,a windows size between succeeding patterns and a
    # maximum size for the number of patterns.
    def abstract_categorical(self, data_table, cols, match, min_support, window_size, max_pattern_size):

        # Find all the relevant columns of binary attributes.
        col_names = list(data_table.columns)
        selected_patterns = []

        relevant_dataset_cols = []
        for i in range(0, len(cols)):
            if match[i] == 'exact':
                relevant_dataset_cols.append(cols[i])
            else:
                relevant_dataset_cols.extend([name for name in col_names if cols[i] in name])

        # Generate the one patterns first
        potential_1_patterns = [[pattern] for pattern in relevant_dataset_cols]

        new_data_table, one_patterns = self.select_k_patterns(data_table, potential_1_patterns, min_support, window_size)
        selected_patterns.extend(one_patterns)
        print(f'Number of patterns of size 1 is {len(one_patterns)}')

        k = 1
        k_patterns = one_patterns

        # And generate all following patterns.
        while (k < max_pattern_size) & (len(k_patterns) > 0):
            k = k + 1
            potential_k_patterns = self.extend_k_patterns(k_patterns, one_patterns)
            new_data_table, selected_new_k_patterns = self.select_k_patterns(new_data_table, potential_k_patterns, min_support, window_size)
            selected_patterns.extend(selected_new_k_patterns)
            print(f'Number of patterns of size {k} is {len(selected_new_k_patterns)}')

        return new_data_table








dataframe_lowpass=dataframe.copy()
LowPass=LowPassFilter()
fs=1000/200
cutoff=1.2

dataframe_lowpass=LowPass.low_pass_filter(dataframe_lowpass,"acc_y",fs,cutoff, order=5)

subset = dataframe_lowpass[dataframe_lowpass["set"] == 45]
print(subset["label"][0])
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot (subset ["acc_y"]. reset_index(drop=True), label="raw data")
ax[1].plot (subset ["acc_y_lowpass"]. reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center",bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

plt.show()

for col in predictor_columns:
    dataframe_lowpass=LowPass.low_pass_filter(dataframe_lowpass,col,fs,cutoff, order=5)
    dataframe_lowpass[col]= dataframe_lowpass[col + "_lowpass"]
    del dataframe_lowpass[col + "_lowpass"]

dataframe_pca=dataframe_lowpass.copy()
PCAA = PrincipalComponentAnalysis()

values_pca=PCAA.determine_pc_explained_variance(dataframe_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), values_pca) 
plt.xlabel("principal component number") 
plt.ylabel("explained variance")
plt. show()
df_pca = PCAA.apply_pca(dataframe_pca, predictor_columns, 3)

subset=dataframe_pca[dataframe_pca["set"]==35]
subset[["pca_1", "pca_2", "pca_3"]].plot()
plt.show()

dataframe_squared=dataframe_pca.copy()

acc_r = dataframe_squared["acc_x"] ** 2 + dataframe_squared["acc_y"] ** 2 + dataframe_squared["acc_z"] ** 2
gyr_r = dataframe_squared["gyr_x"] ** 2 + dataframe_squared["gyr_y"] ** 2 + dataframe_squared["gyr_z"] ** 2
dataframe_squared["acc_r"]=np.sqrt(acc_r)
dataframe_squared["gyr_r"]= np.sqrt(gyr_r)
subset = dataframe_squared [dataframe_squared ["set"] == 14]
subset [["acc_r", "gyr_r"]].plot(subplots=True)
plt.show()

dataframe_temporal=dataframe_squared.copy()
NumAbs=NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

ws = int(1000 / 200)
# Apply temporal abstraction for mean and std across all predictor columns
# First apply the temporal abstraction to the whole dataframe
for col in predictor_columns:
    dataframe_temporal = NumAbs.abstract_numerical(dataframe_temporal, [col], ws, "mean")
    dataframe_temporal = NumAbs.abstract_numerical(dataframe_temporal, [col], ws, "std")

# List to hold the subsets after applying the temporal abstraction
dataframe_temporal_list = []

# Iterate through each unique set and apply temporal abstraction again within subsets
for s in dataframe_temporal["set"].unique():
    # Create a subset of the dataframe based on the "set" column
    subset = dataframe_temporal[dataframe_temporal["set"] == s].copy()
    
    # Apply temporal abstraction for mean and std to each predictor column in the subset
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    
    # Append the processed subset to the list
    dataframe_temporal_list.append(subset)

# Concatenate all subsets back into a single DataFrame
dataframe_temporal = pd.concat(dataframe_temporal_list)

# Display a portion of the DataFrame with specific columns
'''try:
    # Ensure that the columns exist before trying to display them
    print(dataframe_temporal[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]])
except KeyError as e:
    print(f"Error: Missing columns. {e}")'''
    
dataframe_temporal.info()

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()

plt.show()

##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

# Updated by Dave Ebbelaar on 06-01-2023

import numpy as np


# This class performs a Fourier transformation on the data to find frequencies that occur
# often and filter noise.
class FourierTransformation:

    # Find the amplitudes of the different frequencies using a fast fourier transformation. Here,
    # the sampling rate expresses the number of samples per second (i.e. Frequency is Hertz of the dataset).
    def find_fft_transformation(self, data, sampling_rate):
        # Create the transformation, this includes the amplitudes of both the real
        # and imaginary part.
        transformation = np.fft.rfft(data, len(data))
        return transformation.real, transformation.imag

    # Get frequencies over a certain window.
    def abstract_frequency(self, data_table, cols, window_size, sampling_rate):

        # Create new columns for the frequency data.
        freqs = np.round((np.fft.rfftfreq(int(window_size)) * sampling_rate), 3)

        for col in cols:
            data_table[col + "_max_freq"] = np.nan
            data_table[col + "_freq_weighted"] = np.nan
            data_table[col + "_pse"] = np.nan
            for freq in freqs:
                data_table[
                    col + "_freq_" + str(freq) + "_Hz_ws_" + str(window_size)
                ] = np.nan

        # Pass over the dataset (we cannot compute it when we do not have enough history)
        # and compute the values.
        for i in range(window_size, len(data_table.index)):
            for col in cols:
                real_ampl, imag_ampl = self.find_fft_transformation(
                    data_table[col].iloc[
                        i - window_size : min(i + 1, len(data_table.index))
                    ],
                    sampling_rate,
                )
                # We only look at the real part in this implementation.
                for j in range(0, len(freqs)):
                    data_table.loc[
                        i, col + "_freq_" + str(freqs[j]) + "_Hz_ws_" + str(window_size)
                    ] = real_ampl[j]
                # And select the dominant frequency. We only consider the positive frequencies for now.

                data_table.loc[i, col + "_max_freq"] = freqs[
                    np.argmax(real_ampl[0 : len(real_ampl)])
                ]
                data_table.loc[i, col + "_freq_weighted"] = float(
                    np.sum(freqs * real_ampl)
                ) / np.sum(real_ampl)
                PSD = np.divide(np.square(real_ampl), float(len(real_ampl)))
                PSD_pdf = np.divide(PSD, np.sum(PSD))
                data_table.loc[i, col + "_pse"] = -np.sum(np.log(PSD_pdf) * PSD_pdf)

        return data_table
    
    
dataframe_freq=dataframe_temporal.copy().reset_index()
'''dataframe_freq.reset_index()'''
FreqAbs= FourierTransformation()

fs= int(1000/200)
ws= int(2800/200)

dataframe_freq= FreqAbs.abstract_frequency(dataframe_freq,["acc_y"], ws, fs)

dataframe_freq.columns

subset=dataframe_freq[dataframe_freq["set"]==15]
subset[["acc_y"]].plot()
subset[[
    "acc_y_max_freq",
    "acc_y_freq_weighted",
    "acc_y_pse",
    "acc_y_freq_1.429_Hz_ws_14",
    "acc_y_freq_2.5_Hz_ws_14"
]
].plot()
plt.show()

dataframe_freq_list=[]

for s in dataframe_freq["set"].unique():
    print(f"Applying Fourier Transformation to set {s}")
    subset=dataframe_freq[dataframe_freq["set"]==s].reset_index(drop=True).copy()
    subset=FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    dataframe_freq_list.append(subset)
    
dataframe_freq=pd.concat(dataframe_freq_list).set_index("epoch (ms)", drop=True)

dataframe_freq=dataframe_freq.dropna()
dataframe_freq=dataframe_freq.iloc[::2]

from sklearn.cluster import KMeans

dataframe_cluster=dataframe_freq.copy()

cluster_columns=["acc_x", "acc_y", "acc_z"]
k_values= range(2,10)
inertias=[]

for k in k_values:
    subset=dataframe_cluster[cluster_columns]
    kmeans=KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels=kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)
    
plt.figure(figsize=(10,10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel ("Sum of squared distances")
plt.show()

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = dataframe_cluster[cluster_columns]
dataframe_cluster["cluster"] = kmeans.fit_predict(subset)

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in dataframe_cluster["cluster"].unique():
    subset = dataframe_cluster[dataframe_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for l in dataframe_cluster["label"].unique():
    subset = dataframe_cluster[dataframe_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

dataframe_cluster.to_pickle("/Users/siddharthsharma/Desktop/tracking-barbell-exercises/data/interim/03_data_features.pkl")
