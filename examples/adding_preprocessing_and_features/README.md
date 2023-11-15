# Adding Preprocessing Steps and Features

Adding preprocessing steps and features extracts is slightly differently than other addon modules. (This is due to the order in which methods are accessed when controlled by an external yaml file.)

## Preprocessing

As of 11/15/2023, we provide four classes of preprocessing.
1. Signal Processing
2. Noise Reduction
3. mne_processing
4. preprocessing_utils

### Signal Processing
Class designed to handle signal processing related tasks such as
- Downsampling
- Filtering
- etc.

Data is stored within the following objects:
- Single channel data vector: ***self.data***
- Sampling Frequency of input data: ***self.fs***

Returns:
- Output single channel data vector

### Noise Reduction
Class designed to handle signal processing related tasks such as
- Z-score Rejection/Interpolation

Data is stored within the following objects:
- Single channel data vector: ***self.data***
- Sampling Frequency of input data: ***self.fs***

Returns:
- Output single channel data vector

### mne_processing
Class designed to work with built-

### New Classes
all classes should be initialized with a single vector of data (stored to ***self.data***) and the sampling frequency (stored to ***self.fs***). If working within pre-existing classes, 

