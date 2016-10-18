#! /bin/python3

import OpenEphys as ep
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import scipy.io.wavfile
import math


def lowpass(data, cutoff, fs, order=5):
    # TODO check dimensions
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype='low', analog=False)
    y = sig.lfilter(b, a, data)
    return y

def save_wav(data, filename, volume=1):
    if volume > 1 or volume < 0:
        raise RuntimeError('save_wav: volume out of range')
    scaled_data = np.int16(data/np.max(np.abs(data)) * volume * 32767)
    scipy.io.wavfile.write(filename, 44100, scaled_data)

def calc_spectrogram(data, Fs, freq_range=None, log=True, log_ref=1):
    # this could be optimized by computing spectrograms for all channels at once
    # but that would make plotting more complicated
    # TODO what are the actual units of Pxx?
    # is the 10*log10 the right way to get dB?

    # resolution = 1024
    resolution = 256
    (freq_bins, time_bins, Pxx) = sig.spectrogram(
        data, fs=Fs,
        nperseg=resolution,
        noverlap=int(resolution/2),
        mode='psd',
        scaling= 'density')
    # print(time_bins)
    # exit(3)
    Pxx = Pxx.transpose()
    if log:
        Pxx = 10*np.log10(Pxx/log_ref)
    Pxx, freq_bins = truncate_to_range(Pxx, freq_bins, freq_range)
    return (freq_bins, time_bins, Pxx)

def plot_spectrogram(axes, Pxx, time_bins, freq_bins,title='',
                     xlabel='Time (s)',ylabel='Frequency (Hz)',zlabel='PSD (dB)'):
    # TODO WHY ISN'T IT FILLING UP THE TIME AXIS??!!
    print(Pxx.shape)
    print(Pxx.transpose().shape)
    # print(time_bins.shape)
    # print(freq_bins.shape)
    # axes.plot(time_bins, Pxx[:, 128])
    # plt.show()
    # exit(3)
    im = axes.imshow(
        Pxx.transpose(),
        origin="lower",
        aspect="auto",
        extent=[time_bins[0],time_bins[-1],freq_bins[0],freq_bins[-1]],
        cmap=plt.cm.gist_heat,
        interpolation="none")

    # cmap=plt.cm.gist_heat, interpolation="hanning")
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    # plt.colorbar(im, ax=axes, orientation='horizontal').set_label(zlabel)

def calc_and_plot_spectrogram(axes, data, t, Fs, freq_range=None, title='',
                              xlabel='Time (s)',ylabel='Frequency (Hz)',
                              zlabel='PSD (dB)'):

    (freq_bins, time_bins, Pxx)=calc_spectrogram(data, Fs, freq_range=freq_range)
    plot_spectrogram(axes, Pxx, time_bins, freq_bins, title='',
                     xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

def plot_quaternion(axes, x_motion, t_motion, title='', xlabel='Time (s)', ylabel='Amplitude',
                    linestyle='solid', marker=None, linewidth=1, xlim=None, ylim=None, color=None, index_range=[None,None]):
    # Plot each sensor, in the first dimension of x_motion
    for i in range(x_motion.shape[0]):
        plot_time(axes,
                  x_motion[i, index_range[0]:index_range[1]],
                  t_motion[index_range[0]:index_range[1]],
                  title=title, xlabel=xlabel, ylabel=ylabel, linestyle=linestyle,
                  marker=marker, linewidth=linewidth, xlim=xlim, ylim=ylim,
                  color=color)


def plot_time(axes, data, t, title='', xlabel='Time (s)', ylabel='Amplitude',
              linestyle='solid', marker=None, linewidth=1, xlim=None, ylim=None, color=None):

    axes.plot(t[:len(data)], data, linestyle=linestyle, marker=marker, linewidth=linewidth,
              color=color)
    # axes.set_linestyle(linestyle)
    # if marker is not None:
    #     axes.set_marker(marker)
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    if not xlim:
        axes.set_xlim([np.min(t), np.max(t)])
    else:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)


def truncate_to_range(x, t, t_range):
    # TODO test!
    # TODO deal with diff x shapes
    # x and t are numpy arrays
    if t_range is None:
        return (x,t)

    new_x = x.copy()
    new_t = t.copy()
    range_indices = [0, t.shape[-1]-1]
    if t_range[0] > t[0]:
        range_indices[0] = np.argmax(t>t_range[0])
    if t_range[1] < t[-1]:
        range_indices[1] = np.argmax(t>t_range[1])
        new_x = new_x[:, range_indices[0]:range_indices[1]]
        new_t = new_t[range_indices[0]:range_indices[1]]
    return (new_x, new_t)

def are_intervals_close(time_array, value):
    return (np.isclose((time_array[1:] - time_array[:-1]), value).all())

def load_openephys_file(folder, filename, Fs_openephys):
    # always constant for openephys format, at least as of now
    SAMPLES_PER_RECORD = 1024
    all = ep.loadContinuous(folder + filename)
    header = all['header']
    sampleRate = int(header['sampleRate'])
    if sampleRate != Fs_openephys:
        raise RuntimeError('load_openephys_file: unexpected sample rate')

    x = all['data']

    # get timestamps for each record
    timestamps = all['timestamps']
    # compute timestamps for each sample - I think this is the right way...
    t = np.array([np.arange(time, time+1024) for time in timestamps]).flatten() / sampleRate
    if not are_intervals_close(t, 1/sampleRate):
        raise RuntimeError('load_openephys_file: timestamps may be wrong')
    return (x,t)

def debounce_discrete_signal(x, min_samples_per_chunk):
    # Remove any bounces that are shorter than min_samples_per_chunk
    # Do not remove a short leading bounce at the very beginning of the array
    start_index = -1
    x_new = x.copy()
    num_bounces_removed = 0
    for i in range(1, len(x_new)-1):
        # print(start_index)
        if x_new[i] != x_new[i-1]:
            # transition!
            if (start_index > 0) and (i - start_index < min_samples_per_chunk):
                x_new[start_index:i] = x_new[i]
                num_bounces_removed += 1
            else:
                start_index = i
    if num_bounces_removed > 0:
        print('debounce_discrete_signal: removed %d bounces' % num_bounces_removed)
    return x_new

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


# a = np.array([ 0, 1, 1, 1, 0, 2, 2, 2, 2, 1,1,1, 3, 3, 3, 0, 0, 0, 0]) * 1.0
# print(a)
# print(debounce_discrete_signal( a , 2, 3))
# exit(3)

def threshold_01(x, threshold):
    hi_indices = x > threshold
    x_new = np.zeros(x.shape)
    x_new[hi_indices] = 1
    return x_new

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

def normalize(x):
    return x/x.max()

# def quaternion_to_euler(wxyz_list):
# # TODO figure out how to calculate this correctly
#     (w,x,y,z) = tuple(wxyz_list)
#     roll  = math.atan2(2*(w*x + y*z), 1-2*(x**2+ y**2))
#     # print(2*(w*y - z*x))
#     pitch = math.asin(2*(w*y - z*x))

#     yaw   = math.atan2(2*(w*z + x*y), 1-2*(y**2 + z**2))
#     return [roll, pitch, yaw]

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

def load_all_eeg(folder, Fs_openephys, all_channels):
    x_eeg_all = []
    last_t_eeg = None
    t_eeg = None
    for chan_num in all_channels:
        (x_eeg, t_eeg) = load_openephys_file(folder, ("100_CH%d_2.continuous" % chan_num), Fs_openephys)
        x_eeg_all.append(x_eeg)
        if (last_t_eeg is not None) and not np.isclose(t_eeg, last_t_eeg).all():
            raise RuntimeError("load_and_preprocess_all_eeg: EEG file timestamps don't match")
        last_t_eeg = t_eeg

    x_eeg_all = np.array(x_eeg_all)
    return (x_eeg_all, t_eeg)

def reference_all_eeg(x_eeg_all):
    # common average reference
    x_eeg_all_copy = x_eeg_all.copy()
    common_avg = np.mean(x_eeg_all_copy, 0)
    x_eeg_all_copy = x_eeg_all_copy - common_avg

    # print(x_eeg_all_copy.shape)
    # print (common_avg.shape)
    return x_eeg_all_copy


def unwrap_quat(quat):
    """ Remove discontinuities from quaternion data, by letting values go above and below the range."""
    # I don't know how quaternions are supposed to work, is this valid?
    range_size = 2**16
    max_jump_size = range_size/2.0
    new_quat = quat.copy()
    for d in range(new_quat.shape[0]): # for w,x,y,z
        for i in range(1, new_quat.shape[1]): # for each sample
            jump = (new_quat[d][i] - new_quat[d][i-1])
            if jump > max_jump_size:
                # huge jump up, shift back down
                new_quat[d][i] -= range_size
            elif jump < -max_jump_size:
                # huge jump down, shift back up
                new_quat[d][i] += range_size
    return new_quat

def calc_mean_onset(x_eeg, t_eeg, mvmt_onsets, seconds_before, seconds_after, Fs_eeg):
    num_samples_before = seconds_before * Fs_eeg
    num_samples_after  = seconds_after  * Fs_eeg
    x_onsets = []
    for onset in mvmt_onsets:
        onset_index = np.searchsorted(t_eeg, onset)
        x_onsets.append(x_eeg[onset_index-num_samples_before : onset_index+num_samples_after])

    x_onsets = np.array(x_onsets)
    print(x_onsets.shape)
    x_mean_onset = np.mean(x_onsets, 0)
    t_mean_onset = (np.arange(len(x_mean_onset)) - num_samples_before) / Fs_eeg # off by one?
    return (x_mean_onset, t_mean_onset)

def plot_mvmt_onset_lines(ax, mvmt_onsets, line_y_coords, Fs_eeg, color='k'):
    for onset in mvmt_onsets:
        ax.plot([onset, onset+1.0/Fs_eeg], line_y_coords, color=color, linestyle='--')

def plot_mean_onsets_all_channels(all_mvmt_onsets, x_eeg_all, t_eeg, all_eeg_channels, Fs_eeg):
    for chan_num in all_eeg_channels:
        # (x_eeg, t_eeg) = load_openephys_file(folder, ("100_CH%d_2.continuous" % chan_num), Fs_eeg)
        x_eeg = x_eeg_all[chan_num, :]
        (x_mean_onset, t_mean_onset) = calc_mean_onset(x_eeg, t_eeg, all_mvmt_onsets, 1,  1, Fs_eeg)
        plot_time(ax, x_mean_onset, t_mean_onset,
                  xlabel='time (s)', ylabel='eeg magnitude',
                  title=('Mean EEG around movement onsets, channel %d' % chan_num))
        # fig2.savefig('fig_onset_all_chans/onsets_chan_%02d.png' % chan_num)
        # ax.cla()
        plt.show()

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

def get_mvmt_onsets():
    """Return manually marked movement onsets"""
    # first mvmt1 is weird?
    # marked where upper blue and purple diverge
    mvmt1_onsets = [31.85, 43.00, 53.65, 63.75, 73.95, 84.5, 95.4, 105.25, 115.05, 124.45, 134.72 ]

    # marked at downwards turning point of upper light blue
    mvmt2_onsets = [35.54, 46.52, 57.18, 66.97, 77.32, 87.95, 98.27, 108.62, 118.25, 127.95, 137.95 ]

    # baseline = [21.0, 29.0]

    all_mvmt_onsets = np.concatenate((mvmt1_onsets, mvmt2_onsets))
    return all_mvmt_onsets

def test_timing():

    filename_chunk_pin1 =  "100_ADC6_2.continuous"
    filename_chunk_pin2 =  "100_ADC7_2.continuous"
    filename_motion =      "motion9-27-16_2.txt"
    folder = "/home/em/new_data/eeg_test_9-27-16/2016-09-27_19-02-40/"
    Fs_openephys = 30000
    all_eeg_channels = range(1,5)

    ##### load eeg data
    (x_eeg_all, t_eeg) = load_all_eeg(folder, Fs_openephys, all_eeg_channels)

    ##### quick and dirty before/after comparisons of common average referencing
    fig1 = plt.figure()
    ax1 = fig1.gca()
    fig2 = plt.figure()
    ax2_1 = fig2.add_subplot(2,1,1)
    ax2_2 = fig2.add_subplot(2,1,2)

    plot_time(ax1, x_eeg_all[3, :Fs_openephys*2], t_eeg[:Fs_openephys*2], xlabel='', ylabel='EEG Amplitude')
    calc_and_plot_spectrogram(ax2_1,
                              x_eeg_all[3, :Fs_openephys*2],
                              t_eeg[:Fs_openephys*2],
                              Fs_openephys)

    x_eeg_all = reference_all_eeg(x_eeg_all)

    plot_time(ax1, x_eeg_all[3, :Fs_openephys*2], t_eeg[:Fs_openephys*2], xlabel='', ylabel='EEG Amplitude')
    calc_and_plot_spectrogram(ax2_2,
                              x_eeg_all[3, :Fs_openephys*2],
                              t_eeg[:Fs_openephys*2],
                              Fs_openephys)


    ##### load motion data
    (x_motion0, x_motion1, x_motion2, t_motion) =  get_motion(folder,
                                                              filename_motion,
                                                              filename_chunk_pin1,
                                                              filename_chunk_pin2,
                                                              Fs_openephys)

    ##### show movement onset times
    mvmt_onsets = get_mvmt_onsets()

    fig = plt.figure()
    ax = fig.gca()
    plot_mvmt_onset_lines(ax, mvmt_onsets, [80000, -20000], Fs_openephys)
    plot_quaternion(ax, x_motion0, t_motion,  xlabel='', ylabel='Hand Orientation ')
    plot_quaternion(ax, x_motion1, t_motion,  xlabel='', ylabel='Forearm Orientation  ')
    plot_quaternion(ax, x_motion2, t_motion,  ylabel='Upper Arm Orientation ')

    plt.show()




test_timing()


