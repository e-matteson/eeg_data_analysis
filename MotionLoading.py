import numpy as np
import scipy.signal as sig
import json
import math
import matplotlib.pyplot as plt

from MyUtilities import *


def get_chunk_nums(x_chunk_pin1, t_chunk_pin1, x_chunk_pin2, t_chunk_pin2):
    voltage_threshold = 2.0
    min_samples_per_chunk = 10

    # sanity check the dimensions
    assert(np.array_equal(t_chunk_pin1, t_chunk_pin2))

    t_chunk = t_chunk_pin1.copy()
    x_chunk = np.zeros(t_chunk.shape)

    # threshold the analog recordings
    x_chunk_pin1_clean = threshold_01(x_chunk_pin1, voltage_threshold)
    x_chunk_pin2_clean = threshold_01(x_chunk_pin2, voltage_threshold)

    # Calculate the chunk nums encoded by the pins
    x_chunk = x_chunk_pin1_clean + 2*x_chunk_pin2_clean

    # Debounce, because the 2 chunk pins don't change simultaneously, and
    #  sometimes the sample falls between the 2 change times.
    x_chunk = debounce_discrete_signal(x_chunk, min_samples_per_chunk)
    # plot_chunk_length_histogram(x_chunk, t_chunk)
    return (x_chunk, t_chunk)

def find_enable_index(x_chunk):
    # TODO use enable signal instead, I didn't record it in the 9-27-16 test
    for index in range(1, len(x_chunk)):
        if x_chunk[index-1] == 0 and x_chunk[index] == 1:
            # success
            print ("found enable index: %d" % index)
            return index
    # fail
    return -1

def get_motion_values(sample_dict_list, sensor_num):
    """Extract quaternion data for 1 sensor, converting hex strings to ints.
    Return 4xN array."""
    # get hex string data for sensor
    x_motion = [d['data'][sensor_num] for d in sample_dict_list]
    # convert hex string to int
    x_motion = [[int(hex_val, 16) for hex_val in sample] for sample in x_motion]
    # convert to numpy array, reshape so quaternion arrays are in first dimension
    x_motion = np.transpose(np.array(x_motion))
    return x_motion

def make_motion_timestamps(x_chunk, t_chunk, enable_index, samples_per_chunk):
    """The motion sample rate is irregular! Return irregular timestamps.
    Assume samples are evenly spaced within each chunk."""
    # find the start of each chunk
    chunk_start_indices = []
    for i in range(enable_index, len(t_chunk)):
        # TODO off-by-1?
        if x_chunk[i] != x_chunk[i-1]:
            chunk_start_indices.append(i)

    # make evenly spaced sample timestamps within each chunk
    sample_indices = []
    for c in range(len(chunk_start_indices)-1):
        start = chunk_start_indices[c]
        end = chunk_start_indices[c+1]
        interval = (end - start)*1.0/samples_per_chunk
        sample_indices.append(np.arange(start, end, interval))

    # convert from indices to seconds
    sample_indices = np.array(sample_indices, dtype=np.int32).flatten()
    t_motion = t_chunk[sample_indices]

    # if len(t_motion) != x_motion.shape[1]:
    #     raise RuntimeError('make_motion_timestamps: something is very wrong')
    return t_motion

def load_motion_file(folder, filename):
    """Return 3 4xN arrays, with quaternion data for each sensor."""
    sample_dict_list = []
    with open(folder + filename, 'r') as f:
        for line in f:
            sample_dict_list.append(json.loads(line))

    if len(sample_dict_list[0]['data']) != 3:
        raise RuntimeError("load_motion_file: expected 3 motion sensors")

    # # unused:
    # x_motion_chunk_nums = [d['chunk'] for d in sample_dict_list]
    # x_motion_sample_nums = [d['sample'] for d in sample_dict_list]

    # find samples per chunk, by checking the sample num before a zero
    for i in range(1, len(sample_dict_list)):
        if int(sample_dict_list[i]['sample']) == 0:
            samples_per_chunk = 1+int(sample_dict_list[i-1]['sample'])
            break

    x_motion0 = get_motion_values(sample_dict_list, 0)
    x_motion1 = get_motion_values(sample_dict_list, 1)
    x_motion2 = get_motion_values(sample_dict_list, 2)

    return (x_motion0, x_motion1, x_motion2, samples_per_chunk)


def plot_chunk_length_histogram(x_chunk, t_chunk):
    interval_lengths = [] # won't include final interval
    start_index = -1
    for i in range(1, len(x_chunk)-1):
        if x_chunk[i] != x_chunk[i-1]:
            # transition!
            if start_index > 0:
                interval_lengths.append(t_chunk[i] - t_chunk[start_index])
            start_index = i
    plt.hist(interval_lengths[2:])
    plt.xlabel("chunk length (s)")
    plt.title("Inconsistent motion sample period: 2016-9-27 test recording")
    plt.show()
    exit(3)


def get_motion(folder, filename_motion, filename_chunk_pin1, filename_chunk_pin2, Fs_openephys):
    """ Load motion, make timestamps from chunk pins, unwrap quaternions."""

    (x_chunk_pin1, t_chunk_pin1) = load_openephys_file(folder, filename_chunk_pin1, Fs_openephys)
    (x_chunk_pin2, t_chunk_pin2) = load_openephys_file(folder, filename_chunk_pin2, Fs_openephys)

    (x_chunk, t_chunk) = get_chunk_nums(x_chunk_pin1, t_chunk_pin1, x_chunk_pin2, t_chunk_pin2)
    enable_index = find_enable_index(x_chunk)

    (x_motion0, x_motion1, x_motion2, samples_per_chunk) = load_motion_file(folder, filename_motion)
    t_motion = make_motion_timestamps(x_chunk, t_chunk, enable_index, samples_per_chunk)

    x_motion0 = unwrap_quat(x_motion0)
    x_motion1 = unwrap_quat(x_motion1)
    x_motion2 = unwrap_quat(x_motion2)
    return (x_motion0, x_motion1, x_motion2, t_motion)
