import sys
import numpy as np
from os import listdir
from pathlib import Path
from scipy.io import wavfile
from librosa.feature import melspectrogram
from sklearn.preprocessing import LabelBinarizer

 
def get_data(esc50_path, FULL_DATA=False):
    """Get spectra, targets and metadata from the given ESC-50 master folder path.
        Inputs:
            esc50_path: Path to the ESC-50 master folder
            FULL_DATA: Flag indicating if the full 50 class set should be used instead of the 10 class
        Outputs:
            X: Mel-log spectra calculated from ESC-50/10 dataset
            y: Onehot encoded target values
            meta: Metadata for the ESC-50/10 dataset
            """
            
    # All necessary paths
    audio_path = esc50_path  / 'audio'
    meta_path = esc50_path / 'meta'
    spectrogram_path = esc50_path / 'mel_log_spectra'    
    
    
    # Set environment
    __set_env(esc50_path)
    

    # Get mel-log spectra
    if (spectrogram_path / 'mel_log_spectra.npy').exists():
        # Load pre-calculated spectra
        X = np.load(str(spectrogram_path / 'mel_log_spectra.npy'))
    else:
        # No pre-calculated spectra found
        X, fs = __read_audio_data(audio_path)
        X = __mellog_spectra(X,fs)
        np.save(str(spectrogram_path / 'mel_log_spectra'), X)
        
    # Load metadata
    meta = np.loadtxt(str(meta_path / 'esc50.csv'), dtype=str, delimiter=',')        
        
    # Choose between ESC-50/10
    if FULL_DATA:
        ids = np.ones_like(meta[1::,4]).astype(bool)
    else:
        ids = meta[1::,4] == 'True'
        meta = meta[np.concatenate(([True],ids)), :]    
    
    # Extract inputs and targets according to the chosen dataset
    y = meta[1:, 3]  # string labels
    y = __encode_labels(y)  # onehot encoded
    X = X[ids, :, :]  # ESC-50/10
    
    return X, y, meta
    

def class_split(X, y, split_size=0.5, seed=None):
    """Split data based on classes
        Inputs:
            X: Spectra for splitting
            y: Corresponding labels
            split_size: The size of the first splitted batch
            seed: Use a known seed for the rng
        Outputs:
            X_1: Splitted spectra with split_size from the original
            X_2: Splitted spectra with 1-split_size from the original
            y_1: Corresponding labels
            y_2: Corresponding labels
            """
    
    # Set seed if specified
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    # Split classes
    class_ids = np.nonzero(np.sum(y, axis=0))[0]
    n_classes = len(class_ids)
    rng.shuffle(class_ids)
    n_1 = int(split_size*n_classes)
    
    classes_1 = class_ids[:n_1]
    classes_2 = class_ids[n_1:]
    
    classes_1_ids = np.where(y[:,classes_1])[0]
    classes_2_ids = np.where(y[:,classes_2])[0]
    
    X_1 = X[classes_1_ids, :, :]
    X_2 = X[classes_2_ids, :, :]
    
    y_1 = y[classes_1_ids, :]
    y_2 = y[classes_2_ids, :]
    
    return X_1, X_2, y_1, y_2


def __set_env(esc50_path):
    """Set up the environmnent"""
    
      # All necessary paths
    audio_path = esc50_path  / 'audio'
    meta_path = esc50_path / 'meta'
    spectrogram_path = esc50_path / 'mel_log_spectra'
    
    # Make sure the ESC-50 path is valid
    if not audio_path.exists() or not meta_path.exists():
        raise RuntimeError("Error: Invalid ESC-50 path, make sure the path directs to the ESC-50 master directory.")
    
    # Check if a spectrogram folder already exists
    if not spectrogram_path.exists(): 
        spectrogram_path.mkdir()
        
    # Create a checkpoint folder if needed
    if not Path('checkpoint').exists():
        Path('checkpoint').mkdir()
        
    # Create a results folder for saving the final results
    if not Path('results').exists():
        Path('results').mkdir()   
        
    # Create a tmp folder for memory management if needed
    if not Path('tmp').exists():
        Path('tmp').mkdir()   
        

def __read_audio_data(audio_path):
    """Read each ESC-50 audio file from the given path"""
    file_names = listdir(audio_path)
    
    # Get sampling frequency and size from the first file
    fs = wavfile.read(str(audio_path / file_names[0]))[0]
    sz = len(wavfile.read(str(audio_path / file_names[0]))[1])
    
    # Pre-allocate memory
    data = np.zeros((len(file_names), sz))
    
    # Read all files
    sys.stdout.write("\nReading audio files...\n")
    for i in range(len(file_names)):
        data[i,:] = wavfile.read(str(audio_path / file_names[i]))[1]  # read file
        __update_progressbar(i/(len(file_names)-1))  # update progress
    sys.stdout.write("\n")
    sys.stdout.flush()    
    
    return data, fs


def __mellog_spectra(data, fs):
    """Calculate mel-log spectra from the given audio data"""
    
    n_samples = data.shape[0]
    spectra = np.zeros((n_samples, 128, 431))
    
    sys.stdout.write("\nCalculating spectra...\n")
    for i in range(n_samples):
        s = melspectrogram(data[i,:], sr=fs)  # mel-spectrogram
        s[s!=0] = np.log(s[s!=0])  # logarithm
        spectra[i,:,:] = s  # add to data
        __update_progressbar(i/(n_samples-1))  # update progress
    sys.stdout.write("\n")
    sys.stdout.flush()
    
    return spectra


def __encode_labels(y):
    """Encode targets into onehot-form. 
    Also save the string-type labels for future decoding"""
    lb = LabelBinarizer()
    lb.fit(y)
    np.save(Path('tmp') / 'classes.npy', lb.classes_)
    return lb.transform(y)


def __update_progressbar(progress):
    """ Display a progress bar with the given amount of progress"""
    
    # The length of the progress bar
    bar_length = 40 
    
    # Make sure progress is float and not over 1
    progress = float(min(progress, 1))
    
    # Number of blocks currently shown
    block = int(round(bar_length*progress))
    
    # Make a progress bar
    text = "\rProgress: [{}] {:4.1f}%".format( "#"*block + "-"*(bar_length-block), progress*100)
                        
    # Display
    sys.stdout.write(text)
    sys.stdout.flush()
