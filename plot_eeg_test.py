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

    Pxx = Pxx.transpose()
    if log:
        Pxx = 10*np.log10(Pxx/log_ref)
    Pxx, freq_bins = truncate_to_range(Pxx, freq_bins, freq_range)
    return (freq_bins, time_bins, Pxx)

def plot_spectrogram(axes, Pxx, t, freq_bins,title='',
                     xlabel='Time (s)',ylabel='Frequency (Hz)',zlabel='PSD (dB)'):

    im = axes.imshow(Pxx.transpose(), origin="lower", aspect="auto",
                     extent=[t[0],t[-1],freq_bins[0],freq_bins[-1]],
                     cmap=plt.cm.gist_heat, interpolation="none")
                     # cmap=plt.cm.gist_heat, interpolation="hanning")
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    plt.colorbar(im, ax=axes, orientation='horizontal').set_label(zlabel)

def calc_and_plot_spectrogram(axes, data, t, Fs, freq_range=None, title='',
                              xlabel='Time (s)',ylabel='Frequency (Hz)',
                              zlabel='PSD (dB)'):

    (freq_bins, time_bins, Pxx)=calc_spectrogram(data, Fs, freq_range=freq_range)
    plot_spectrogram(axes, Pxx, time_bins, freq_bins, title='',
                     xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

def plot_quaternion(axes, x_motion, t_motion, title='', xlabel='Time (s)', ylabel='Amplitude',
                    linestyle='solid', marker=None, linewidth=1, xlim=None, ylim=None, color=None, index_range=None):
    if(not index_range):
        plot_time(axes, x_motion[0, :], t_motion, title=title, xlabel=xlabel,
                  ylabel=ylabel, linestyle=linestyle, marker=marker,
                  linewidth=linewidth, xlim=xlim, ylim=ylim, color=color)

        plot_time(axes, x_motion[1, :], t_motion, title=title, xlabel=xlabel,
                  ylabel=ylabel, linestyle=linestyle, marker=marker,
                  linewidth=linewidth, xlim=xlim, ylim=ylim, color=color)

        plot_time(axes, x_motion[2, :], t_motion, title=title, xlabel=xlabel,
                  ylabel=ylabel, linestyle=linestyle, marker=marker,
                  linewidth=linewidth, xlim=xlim, ylim=ylim, color=color)

        plot_time(axes, x_motion[3, :], t_motion, title=title, xlabel=xlabel,
                  ylabel=ylabel, linestyle=linestyle, marker=marker,
                  linewidth=linewidth, xlim=xlim, ylim=ylim, color=color)

    else:
        plot_time(axes, x_motion[0, index_range[0]:index_range[1]],
                  t_motion[index_range[0]:index_range[1]], title=title, xlabel=xlabel,
                  ylabel=ylabel, linestyle=linestyle, marker=marker,
                  linewidth=linewidth, xlim=xlim, ylim=ylim, color=color)

        plot_time(axes, x_motion[1, index_range[0]:index_range[1]],
                  t_motion[index_range[0]:index_range[1]], title=title, xlabel=xlabel,
                  ylabel=ylabel, linestyle=linestyle, marker=marker,
                  linewidth=linewidth, xlim=xlim, ylim=ylim, color=color)

        plot_time(axes, x_motion[2, index_range[0]:index_range[1]],
                  t_motion[index_range[0]:index_range[1]], title=title, xlabel=xlabel,
                  ylabel=ylabel, linestyle=linestyle, marker=marker,
                  linewidth=linewidth, xlim=xlim, ylim=ylim, color=color)

        plot_time(axes, x_motion[3, index_range[0]:index_range[1]],
                  t_motion[index_range[0]:index_range[1]], title=title, xlabel=xlabel,
                  ylabel=ylabel, linestyle=linestyle, marker=marker,
                  linewidth=linewidth, xlim=xlim, ylim=ylim, color=color)


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

def load_openephys(folder, filename, Fs_openephys):
    # always constant for openephys format, at least as of now
    SAMPLES_PER_RECORD = 1024
    all = ep.loadContinuous(folder + filename)
    header = all['header']
    sampleRate = int(header['sampleRate'])
    if sampleRate != Fs_openephys:
        raise RuntimeError('load_openephys: unexpected sample rate')

    x = all['data']

    # get timestamps for each record
    timestamps = all['timestamps']
    # compute timestamps for each sample - I think this is the right way...
    t = np.array([np.arange(time, time+1024) for time in timestamps]).flatten() / sampleRate
    if not are_intervals_close(t, 1/sampleRate):
        raise RuntimeError('load_openephys: timestamps may be wrong')
    return (x,t)

# def threshold_debounce(x, threshold, min_samples_per_chunk):
#     # Threshold the signal to 1 and 0
#     # Remove any high/low periods that are shorter than min_samples_per_chunk
#     # Do not remove a short leading period at the very beginning of the array
#     hi_indices = x > threshold

#     x_new = np.zeros(x.shape)
#     x_new[hi_indices] = 1
#     start_index = -1
#     last_val = x_new[0]
#     for i in range(1, len(x_new)-1):
#         # print(start_index)
#         if x_new[i] != x_new[i-1]:
#             # transition!
#             if (start_index > 0) and (i - start_index < min_samples_per_chunk):
#                 x_new[start_index:i] = x_new[i]
#             else:
#                 start_index = i
#     return x_new

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
    plot_chunk_length_histogram(x_chunk, t_chunk)
    return (x_chunk, t_chunk)

def find_enable_index(x_chunk, t_chunk):
    # TODO use enable signal instead, I didn't record it in the 9-27-16 test
    for index in range(1, len(x_chunk)):
        if x_chunk[index-1] == 0 and x_chunk[index] == 1:
            # success
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
    x_motion = [d['data'][sensor_num] for d in sample_dict_list]
    x_motion = [[int(hex_val, 16) for hex_val in sample] for sample in x_motion]
    x_motion = np.array(x_motion)
    x_motion = np.transpose(x_motion)
    return x_motion

def load_motion(folder, filename, Fs_openephys, Fs_motion):
    sample_dict_list = []
    with open(folder + filename, 'r') as f:
        for line in f:
            sample_dict_list.append(json.loads(line))

    x_motion_chunk_nums = [d['chunk'] for d in sample_dict_list]
    x_motion_sample_nums = [d['sample'] for d in sample_dict_list]

    x_motion0 = get_motion_values(sample_dict_list, 0)
    x_motion1 = get_motion_values(sample_dict_list, 1)
    x_motion2 = get_motion_values(sample_dict_list, 2)
    t_motion = np.array(np.arange(x_motion0.shape[1])) / Fs_motion

    # TODO WARNING the openephys and motion times are off by a factor of 2,
    # and this is a super shitty hack to make them line up, probably,
    # with out fixing or understanding the underlying problem!!!!!!!!

    # t_motion = t_motion * 2
    # new_num_samples = len(t_motion)* 1.0 * Fs_openephys / Fs_motion
    # (x_motion0, t_motion) = sig.resample(x_motion0, new_num_samples, t=t_motion, axis=0, window='hanning')

    print(x_motion0.shape)
    print(t_motion.shape)

    return (x_motion0, x_motion1, x_motion2, t_motion)

def make_bad_plots():
    folder = "/home/em/new_data/eeg_test_9-27-16/2016-09-27_19-02-40/"
    Fs_openephys = 30000
    Fs_motion = 100 # motion sample rate in Hz

    (x_audio, t_audio) =   load_openephys(folder, "100_ADC5_2.continuous", Fs_openephys)
    (x_chunk_pin1, t_chunk_pin1) = load_openephys(folder, "100_ADC6_2.continuous", Fs_openephys)
    (x_chunk_pin2, t_chunk_pin2) = load_openephys(folder, "100_ADC7_2.continuous", Fs_openephys)
    (x_chunk, t_chunk) = get_chunk_nums(x_chunk_pin1, t_chunk_pin1, x_chunk_pin2, t_chunk_pin2)
    (x_chan4, t_chan4) = load_openephys(folder, "100_CH4_2.continuous", Fs_openephys)

    (x_motion0, x_motion1, x_motion2, t_motion) = load_motion(folder, "motion9-27-16_2.txt", Fs_openephys, Fs_motion)

    # x_motion0 = normalize(x_motion0)
    # x_motion1 = normalize(x_motion1)
    # x_motion2 = normalize(x_motion2)
    # x_chan4 = normalize(x_chan4)
    # x_audio = normalize(x_audio)

    motion_enable_index = find_enable_index(x_chunk, t_chunk)
    # offset the motion times so it matches the start of the openephys recording
    t_motion = t_motion + t_audio[motion_enable_index]

    # truncate the openephys data so it starts when the motion recording was enabled
    # # index_range = [motion_enable_index + Fs_openephys*2, motion_enable_index + Fs_openephys*5]
    # index_range = [motion_enable_index, motion_enable_index + Fs_openephys*20]
    index_range = [motion_enable_index, -1]
    motion_index_range = [math.floor(i*1.0 / Fs_openephys * Fs_motion) for i in index_range]


    x_chunk = x_chunk[index_range[0] : index_range[1]]
    t_chunk = t_chunk[index_range[0] : index_range[1]]
    x_chan4 = x_chan4[index_range[0] : index_range[1]]
    t_chan4 = t_chan4[index_range[0] : index_range[1]]
    x_audio = x_audio[index_range[0] : index_range[1]]
    t_audio = t_audio[index_range[0] : index_range[1]]


    # t_motion = t_motion[index_range[0] : index_range[1]]
    # t_motion = t_motion[motion_index_range[0] : motion_index_range[1]]
    # x_motion0 = x_motion0[motion_index_range[0] : motion_index_range[1], :]
    # x_motion1 = x_motion1[motion_index_range[0] : motion_index_range[1], :]
    # x_motion2 = x_motion2[motion_index_range[0] : motion_index_range[1], :]
    print(x_motion0.shape)
    print(t_motion.shape)

    fig = plt.figure()
    ax1 = fig.add_subplot(5,1,1)
    ax2 = fig.add_subplot(5,1,2)
    ax3 = fig.add_subplot(5,1,3)
    ax4 = fig.add_subplot(5,1,4)
    ax5 = fig.add_subplot(5,1,5)
    t_range = [48, 58]
    # t_range = [48, 300]
    x_audio_low = lowpass(x_audio, 5000, Fs_openephys)
    plot_time(ax1, x_chan4, t_chan4, xlim=t_range, xlabel='', ylabel='EEG Amplitude')
    plot_time(ax2, x_audio_low, t_audio, ylim=[0.01, 0.07], xlim=t_range, xlabel='', ylabel='Audio Volume')
    plot_quaternion(ax3, x_motion0, t_motion, xlim=t_range, xlabel='', ylabel='Hand Orientation ')
    plot_quaternion(ax4, x_motion1, t_motion, xlim=t_range, xlabel='', ylabel='Forearm Orientation  ')
    plot_quaternion(ax5, x_motion2, t_motion, xlim=t_range, ylabel='Upper Arm Orientation ')

    # fig2 = plt.figure()
    # # calc_and_plot_spectrogram(fig2.gca(), x_audio, t_audio, Fs_openephys)
    # calc_and_plot_spectrogram(fig2.gca(), x_audio, t_audio, Fs_openephys)

    # ax.plot(xlow)
    plt.show()

    # plt.savefig("figs/crappy_synch_plot.png", bbox_inches='tight')

    # save_wav(xlow, "audio_test.wav")


def test_timing():
    # ax1 = fig.add_subplot(5,1,1)
    fig = plt.figure()
    ax1 = fig.gca()


    folder = "/home/em/new_data/eeg_test_9-27-16/2016-09-27_19-02-40/"

    Fs_openephys = 30000
    Fs_motion = 100 # motion sample rate in Hz

    (x_chunk_pin1, t_chunk_pin1) = load_openephys(folder, "100_ADC6_2.continuous", Fs_openephys)

    # plt.hist(tdiffs, bins=1000)
    # plot_time(ax1, tdiffs, range(len(tdiffs)), xlabel='', ylabel='chunk 1')
    # plot_time(ax1, t_chunk_pin1, range(len(t_chunk_pin1)), xlabel='', ylabel='chunk 1')

    (x_chunk_pin2, t_chunk_pin2) = load_openephys(folder, "100_ADC7_2.continuous", Fs_openephys)
    (x_chunk, t_chunk) = get_chunk_nums(x_chunk_pin1, t_chunk_pin1, x_chunk_pin2, t_chunk_pin2)
    # plot_time(ax1, x_chunk_pin1, t_chunk_pin1, xlabel='', ylabel='chunk nums')
    plot_time(ax1, x_chunk, t_chunk, xlabel='', ylabel='chunk nums')
    plt.show()

    exit(3)
    plot_time(ax1, x_chunk_pin1, t_chunk_pin1, xlabel='', ylabel='chunk 1')
    plot_time(ax1, x_chunk_pin2, t_chunk_pin2, xlabel='', ylabel='chunk 2')

    # plot_quaternion(ax3, x_motion0, t_motion, xlim=t_range, xlabel='', ylabel='Hand Orientation ')
    # plot_quaternion(ax4, x_motion1, t_motion, xlim=t_range, xlabel='', ylabel='Forearm Orientation  ')
    # plot_quaternion(ax5, x_motion2, t_motion, xlim=t_range, ylabel='Upper Arm Orientation ')
    plt.show()

    # (x_motion0, x_motion1, x_motion2, t_motion) = load_motion(folder, "motion9-27-16_2.txt", Fs_openephys, Fs_motion)

    # print(x_motion0.shape)
    # print(t_motion.shape)




test_timing()

# def main():
# main()
