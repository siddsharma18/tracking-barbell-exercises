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