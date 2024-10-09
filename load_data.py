import pickle
import numpy as np
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar_batch(filename):
    # Load a single batch of the CIFAR-10 dataset
    try:
        # Use the unpickle function to load the batch
        batch = unpickle(filename)
        # Extract data (images) and labels from the batch
        data = batch[b'data']
        labels = batch[b'labels']
        print(f'data shape: {data.shape}, labels length: {len(labels)}')
    except FileNotFoundError:
        # Raise an error if the file is not found
        raise FileNotFoundError(f"The file {filename} was not found.")
    return data, labels

def load_all_batches():
    # Load all the batches of the CIFAR-10 dataset
    data_list = []
    labels_list = []
    for i in range(1, 6):
        # Load each of the 5 data batches
        print(f'Loading batch {i}')
        base_path = 'cifar-10-batches-py/'
        data, labels = load_cifar_batch(f'{base_path}data_batch_{i}')
        # Append the data and labels to the lists
        data_list.append(data)
        labels_list.append(labels)

    # Concatenate all data batches into a single array
    all_data = np.concatenate(data_list, axis=0)
    # Concatenate all label batches into a single array
    all_labels = np.concatenate(labels_list, axis=0)
    print(f'All data shape: {all_data.shape}, All labels shape: {all_labels.shape}')
    return all_data, all_labels

def normalize_data(data):
    # Normalize the data to have values between -0.5 and 0.5
    data = data.astype('float32') / 255.0  # Scale pixel values to [0, 1]
    data = data - 0.5  # Shift pixel values to [-0.5, 0.5]
    return data

def load_data():
    # Load CIFAR-10 dataset
    print('Loading CIFAR-10 dataset...')
    cifar_data, cifar_labels = load_all_batches()

    # Normalize CIFAR-10 data
    normalized_data = normalize_data(cifar_data)
    print(f'Labels shape: {cifar_labels.shape}')
    return normalized_data, cifar_labels