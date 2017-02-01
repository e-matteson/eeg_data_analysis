#! /bin/python3

import OpenEphys as ep
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.signal as sig
import scipy.stats as stats


from MyUtilities import *
from MotionLoading import *


def preprocess_eeg(x_eeg_all, t_eeg, eeg_lowpass_cutoff, eeg_downsample_factor, Fs_openephys, use_CAR=True):
    ##### quick and dirty before/after comparisons of common average referencing
    # fig1 = plt.figure()
    # ax1 = fig1.gca()
    # fig2 = plt.figure()
    # ax2_1 = fig2.add_subplot(2,1,1)
    # ax2_2 = fig2.add_subplot(2,1,2)

    # plot_time(ax1, x_eeg_all[3, :Fs_openephys*2], t_eeg[:Fs_openephys*2], xlabel='', ylabel='EEG Amplitude')
    # calc_and_plot_spectrogram(ax2_1,
    #                           x_eeg_all[3, :Fs_openephys*2],
    #                           t_eeg[:Fs_openephys*2],
    #                           Fs_openephys)

    if use_CAR:
        x_eeg_all = reference_all_eeg(x_eeg_all)

    x_eeg_all = lowpass_all_eeg(x_eeg_all, eeg_lowpass_cutoff, Fs_openephys)

    Fs_eeg = Fs_openephys / eeg_downsample_factor
    (x_eeg_all, t_eeg) = downsample_all_eeg(x_eeg_all, t_eeg, eeg_downsample_factor)

    # plot_time(ax1, x_eeg_all[3, :Fs_eeg*2], t_eeg[:Fs_eeg*2], xlabel='', ylabel='EEG Amplitude')
    # calc_and_plot_spectrogram(ax2_2,
    #                           x_eeg_all[3, :Fs_eeg*2],
    #                           t_eeg[:Fs_eeg*2],
    #                           Fs_eeg)

    return (x_eeg_all, t_eeg, Fs_eeg)

def reference_all_eeg(x_eeg_all):
    """Return common average referenced EEG channels"""
    # common average reference
    x_eeg_all_copy = x_eeg_all.copy()
    common_avg = np.mean(x_eeg_all_copy, 0)
    x_eeg_all_copy = x_eeg_all_copy - common_avg

    # print(x_eeg_all_copy.shape)
    # print (common_avg.shape)
    return x_eeg_all_copy


def downsample_all_eeg(x_eeg_all, t_eeg, factor):
    """Downsample all the eeg data, and the time.
    Does not include an anti-aliasing filter!!!"""
    t_eeg_new = downsample(t_eeg, factor)
    x_eeg_all_new = []
    for chan_index in range(x_eeg_all.shape[0]):
        x_eeg_all_new.append(downsample(x_eeg_all[chan_index], factor))

    return (np.array(x_eeg_all_new), t_eeg_new)

def lowpass_all_eeg(x_eeg_all, cutoff, Fs_eeg):
    x_eeg_all_copy = np.zeros(x_eeg_all.shape)
    for chan_index in range(x_eeg_all.shape[0]):
        x_eeg_all_copy[chan_index, :] = lowpass(x_eeg_all[chan_index, :], cutoff, Fs_eeg)
    return x_eeg_all_copy

def get_mvmt_onsets():
    """Return manually marked movement onset times, in seconds"""
    # # first mvmt1 is weird?
    # # marked where upper blue and purple diverge
    # mvmt1_onsets = [31.85, 43.00, 53.65, 63.75, 73.95, 84.5, 95.4, 105.25, 115.05, 124.45, 134.72 ]

    # # marked at downwards turning point of upper light blue
    # mvmt2_onsets = [35.54, 46.52, 57.18, 66.97, 77.32, 87.95, 98.27, 108.62, 118.25, 127.95, 137.95 ]

    # baseline_interval = [21.0, 29.0]

    all_mvmt_onsets = np.concatenate((mvmt1_onsets, mvmt2_onsets))
    return (all_mvmt_onsets, baseline_interval)

def plot_mvmt_onset_lines(ax, mvmt_onsets, line_y_coords, Fs_eeg, color='k'):
    for onset in mvmt_onsets:
        ax.plot([onset, onset+1.0/Fs_eeg], line_y_coords, color=color, linestyle='--')

def show_mvmt_onset_lines_over_quats(mvmt_onsets, x_motion0, x_motion1, x_motion2, t_motion, Fs_eeg):
    fig = plt.figure()
    ax = fig.gca()
    plot_mvmt_onset_lines(ax, mvmt_onsets, [80000, -20000], Fs_eeg)
    plot_quaternion(ax, x_motion0, t_motion,  xlabel='', ylabel='Hand Orientation ')
    plot_quaternion(ax, x_motion1, t_motion,  xlabel='', ylabel='Forearm Orientation  ')
    plot_quaternion(ax, x_motion2, t_motion,  ylabel='Upper Arm Orientation ')
    plt.show()

def calc_mean_onset_LMP(x_eeg, t_eeg, mvmt_onsets, time_interval, Fs_eeg):
    """Return 1 eeg channel time-domain averaged over all mvmt_onset times.
    time_interval is [seconds_before_onset, seconds_after_onset]"""
    num_samples_before = time_interval[0] * Fs_eeg
    num_samples_after  = time_interval[1] * Fs_eeg
    x_onsets = []
    for onset in mvmt_onsets:
        onset_index = np.searchsorted(t_eeg, onset)
        x_onsets.append(x_eeg[onset_index-num_samples_before : onset_index+num_samples_after])

    x_onsets = np.array(x_onsets)
    x_mean_onset = np.mean(x_onsets, axis=0)
    x_sem_onset = stats.sem(x_onsets, axis=0)
    # x_std_onset = np.std(x_onsets, axis=0)
    t_mean_onset = (np.arange(len(x_mean_onset)) - num_samples_before) / Fs_eeg # off by one?
    return (x_mean_onset, x_sem_onset, t_mean_onset, x_onsets)

def plot_mean_onset_LMP_all_channels(x_eeg_all, t_eeg, all_mvmt_onsets, time_interval, all_eeg_channels, Fs_eeg):
    fig = plt.figure()
    ax = fig.gca()
    for chan_name in all_eeg_channels:
        chan_index = all_eeg_channels.index(chan_name)
        print(chan_index)
        x_eeg = x_eeg_all[chan_index, :]
        # get mean and standard error
        (x_mean_onset, x_sem_onset, t_mean_onset, x_onsets) = calc_mean_onset_LMP(x_eeg, t_eeg, all_mvmt_onsets, time_interval, Fs_eeg)

        #### plot all
        # for onset in range(len(all_mvmt_onsets)):
        #     plot_time(ax, x_onsets[onset], t_mean_onset)

        ##### plot mean
        plot_time(ax, x_mean_onset, t_mean_onset,
                  linewidth='2',
                  xlabel='time (s)', ylabel='eeg magnitude',
                  title=('Mean EEG around movement onsets, SEM, channel %d' % chan_name))

        # plot standard error
        ax.fill_between(t_mean_onset,
                        x_mean_onset - x_sem_onset,
                        x_mean_onset + x_sem_onset,
                        color='grey')

        fig.savefig('fig_onset_all_chans_downsample/onsets_chan_%02d.png' % chan_name)
        ax.cla()

def calc_mean_baseline_power(x_eeg, t_eeg, baseline_interval, freq_index_interval, Fs_eeg):
    """Return scalar value, baseline power for 1 channel"""
    x_baseline_powers = []
    baseline_indices = [np.searchsorted(t_eeg, baseline_interval[0]), np.searchsorted(t_eeg, baseline_interval[1])]
    x_baseline = x_eeg[baseline_indices[0] : baseline_indices[1]]
    (freq_bins, time_bins, Pxx) = calc_spectrogram(x_baseline, Fs_eeg,
                                                   freq_range=[0,100],
                                                   log=True)
    print(freq_bins)
    band_power = np.sum(Pxx[:, freq_index_interval[0]:freq_index_interval[1]], axis=1)
    x_baseline_powers.append(band_power)
    x_baseline_powers = np.array(x_baseline_powers)

    x_mean_baseline_power = np.mean(np.mean(x_baseline_powers, axis=0), axis=0)
    # x_sem_baseline_power = stats.sem(x_baseline_powers, axis=0)
    # x_std_baseline_power = np.std(x_baseline_powers, axis=0)
    freq_hz_interval = [freq_bins[freq_index_interval[0]], freq_bins[freq_index_interval[1]]]
    print (freq_hz_interval)
    return (x_mean_baseline_power, freq_hz_interval)

def get_baseline_power_all_channels(x_eeg_all, t_eeg, baseline_interval, freq_interval, all_eeg_channels, Fs_eeg):
    ####### get baseline power
    baseline_power_all_channels = []
    for chan_name in all_eeg_channels:
        chan_index = all_eeg_channels.index(chan_name)
        x_eeg = x_eeg_all[chan_index]
        (x_mean_baseline_power, freq_hz_interval) = calc_mean_baseline_power(x_eeg, t_eeg, baseline_interval, freq_interval, Fs_eeg)
        baseline_power_all_channels.append(x_mean_baseline_power)

    return np.mean(baseline_power_all_channels)

def calc_mean_onset_power(x_eeg, t_eeg, mvmt_onsets, time_interval, freq_index_interval, Fs_eeg):
    """Return 1 eeg channel freq-domain averaged over all mvmt_onset times.
    time_interval is [seconds_before_onset, seconds_after_onset]"""
    num_samples_before = time_interval[0] * Fs_eeg
    num_samples_after  = time_interval[1] * Fs_eeg
    x_onset_powers = []
    for onset in mvmt_onsets:
        onset_index = np.searchsorted(t_eeg, onset)
        x_onset = x_eeg[onset_index-num_samples_before : onset_index+num_samples_after]
        (freq_bins, time_bins, Pxx) = calc_spectrogram(x_onset, Fs_eeg,
                                                             freq_range=[0,100],
                                                             log=True)
        # print(freq_bins)
        band_power = np.sum(Pxx[:, freq_index_interval[0]:freq_index_interval[1]], axis=1)
        x_onset_powers.append(band_power)

    x_onset_powers = np.array(x_onset_powers)

    x_mean_onset_power = np.mean(x_onset_powers, axis=0)
    x_sem_onset_power = stats.sem(x_onset_powers, axis=0)
    # x_std_onset_power = np.std(x_onset_powers, axis=0)
    t_mean_onset_power = time_bins - sum(time_interval)/2.0
    freq_hz_interval = [freq_bins[freq_index_interval[0]], freq_bins[freq_index_interval[1]]]
    return (x_mean_onset_power, x_sem_onset_power, t_mean_onset_power, x_onset_powers, freq_hz_interval)


def plot_mean_onset_power_all_channels(x_eeg_all, t_eeg, all_mvmt_onsets, time_interval,
                                       freq_index_interval, all_eeg_channels, Fs_eeg,baseline_power=1):
    fig = plt.figure()
    ax = fig.gca()
    print(all_eeg_channels)
    for chan_name in all_eeg_channels:
        chan_index = all_eeg_channels.index(chan_name)
        print((chan_name, chan_index))
        x_eeg = x_eeg_all[chan_index, :]

        # get mean and standard error / deviation
        (x_mean_onset_power, x_sem_onset_power, t_mean_onset_power,
         x_onset_powers, freq_hz_interval) = calc_mean_onset_power(
             x_eeg, t_eeg, all_mvmt_onsets, time_interval, freq_index_interval, Fs_eeg)
        #### plot all
        # for onset in range(len(all_mvmt_onsets)):
        #     plot_time(ax, x_onsets[onset], t_mean_onset)

        ##### plot mean
        title_str = ('EEG power in [%.1f - %.1f Hz] band averaged across\n %d movement onsets (channel %d)'
                     % (freq_hz_interval[0], freq_hz_interval[1], len(all_mvmt_onsets), chan_name))
        plot_time(ax, toDecibels(x_mean_onset_power, baseline_power), t_mean_onset_power,
                  linewidth='4',
                  title=title_str,
                  color='black')

        sem_high =       toDecibels(x_mean_onset_power + x_sem_onset_power, baseline_power)
        sem_low  =       toDecibels(x_mean_onset_power - x_sem_onset_power, baseline_power)
        # plot standard error
        ax.fill_between(t_mean_onset_power, sem_low, sem_high, color='grey')

        # ax.set_xlim([np.min(t_mean_onset_power), np.max(t_mean_onset_power)])
        # ax.set_ylim([np.min(sem_low)*1.1, np.max(sem_high)*1.1])

        ax.legend(["mean", "SEM"],
                  bbox_to_anchor=(.32, .25),
                  bbox_transform=plt.gcf().transFigure,
                  frameon=False)

        ylim = ax.get_ylim()
        ax.plot([0,0], ylim, '--', color='black', linewidth=5)

        ax.set_xlabel('Time from movement onset [s]')
        ax.set_ylabel('Power [dB]')

        # plt.show()
        # exit(3)
        fig.savefig('fig_onset_power/f%d_%d_onsets_chan_%02d.png'
                    % (freq_index_interval[0], freq_index_interval[1], chan_name))
        ax.cla()

def plot_power_all_channels(x_eeg_all, t_eeg, freq_hz_interval, all_eeg_channels,
                            Fs_eeg, baseline_power=1, titlestr='', index=0):
    print(all_eeg_channels)
    for chan_name in all_eeg_channels:
        fig = plt.figure()
        ax = fig.gca()
        chan_index = all_eeg_channels.index(chan_name)
        x_eeg = x_eeg_all[chan_index, :]
        calc_and_plot_spectrogram(ax, x_eeg, t_eeg, Fs_eeg,
                                  freq_range=freq_hz_interval,
                                  log=True, log_ref=baseline_power,
                                  xlabel='Time (s)',ylabel='Frequency (Hz)',
                                  zlabel='PSD (dB)',
                                  colorbar=True,
                                  vmin=-30, vmax=10,
                                  title=titlestr + (' channel %d' % chan_name))

        filename =('fig_spectrogram_low/spectrogram_i%d_chan_%02d.png' % (index, chan_name))
        fig.savefig(filename )
        fig.clear()

def plot_quick_summary(all_eeg_channels, x_eeg_all, t_eeg, Fs_eeg, Fs_openephys,
                       eeg_downsample_factor, eeg_lowpass_cutoff,
                       data_session_name, figure_directory, use_CAR=True):
    ##### make figure directory
    # make_directory(figure_directory)

    ##### plot all eeg, time and spectrogram
    fig = plt.figure()
    ax = fig.gca()

    ### plot raw eeg
    for chan_name in all_eeg_channels:
        title_str = ('%s, Fs=%dHz, channel %d' % (data_session_name, Fs_eeg, chan_name))
        chan_index = all_eeg_channels.index(chan_name)
        calc_and_plot_spectrogram(ax,
                                  x_eeg_all[chan_index],
                                  t_eeg,
                                  Fs_eeg,
                                  title=title_str)
        fig.savefig(figure_directory + '/chan_%02d_raw.png' % chan_name)
        ax.cla()

    ### filter and downsample eeg
    (x_eeg_all, t_eeg, Fs_eeg) = preprocess_eeg(x_eeg_all, t_eeg, eeg_lowpass_cutoff, eeg_downsample_factor, Fs_openephys, use_CAR=use_CAR)

    ### plot filtered eeg
    for chan_name in all_eeg_channels:
        chan_index = all_eeg_channels.index(chan_name)

        title_str = ('%s, Fs=%d, lowpass %d, channel %d' % (data_session_name, Fs_eeg, eeg_lowpass_cutoff, chan_name))
        plot_time(ax, x_eeg_all[chan_index, :], t_eeg,
                  ylabel='EEG Magnitude',
                  title=title_str)
        fig.savefig(figure_directory + '/chan_%02d_time.png' % chan_name)

        ax.cla()
        calc_and_plot_spectrogram(ax,
                                  x_eeg_all[chan_index],
                                  t_eeg,
                                  Fs_eeg,
                                  freq_range=[0, eeg_lowpass_cutoff],
                                  title=title_str)
        fig.savefig(figure_directory + '/chan_%02d_freq.png' % chan_name)
        ax.cla()

def plot_rms_noise_comparisons(Fs_openephys, Fs_eeg, directories):
    fig = plt.figure()

    high_cutoff = 1000
    chan_name = 8
    chan_index = 0

    for data_directory in directories:
        (x_eeg_all, t_eeg) = load_eeg(data_directory, Fs_openephys, [chan_name], 1)

        ##### plot all eeg, time and spectrogram
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)

        ### get eeg power
        x = x_eeg_all[chan_index]

        x_high = highpass(x, high_cutoff, Fs_eeg)
        rms = moving_RMS(x, 256)
        rms_high = moving_RMS(x_high, 256)


        ax1.plot(t_eeg, rms)
        ax2.plot(t_eeg, rms_high)
        ax1.set_title("a) Unfiltered recordings")
        ax2.set_title("b) Highpassed recordings, %dHz" % high_cutoff)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("moving RMS (uV)")
        # fig.suptitle("Comparison of noise power in two EMG recordings, with different filters. In (a), the low frequency EMG signals are much stronger than the accelerometer noise, for both wall-power and battery-power recordings. \nIn (b), both the EMG recordings are highpass filtered to reveal accelerometer noise. The noise is reduced when the beaglebone is electrically isolated and battery-powered. \n\n Commit _______, channel 8, '2016-11-21_18-37-28' (wall) & '2016-11-21_18-31-56' (battery)")


        ymin1 = ax1.get_ylim()[0]
        height1 = ax1.get_ylim()[1] - ax1.get_ylim()[0]
        ymin2 = ax2.get_ylim()[0]
        height2 = ax2.get_ylim()[1] - ax2.get_ylim()[0]

        acq_x_min = 21
        acq_x_width = 25
        acq_color = 'yellow'
        ax1.add_patch(patches.Rectangle( (acq_x_min, ymin1),
                                            acq_x_width, height1,
                                            facecolor=acq_color,
                                            edgecolor='none',
                                            alpha=0.2
        ))
        ax2.add_patch(patches.Rectangle( (acq_x_min, ymin2),
                                            acq_x_width, height2,
                                            facecolor=acq_color,
                                            edgecolor='none',
                                            alpha=0.2
        ))


        box_x_mins = [6, 16, 26, 36]
        box_w_widths = [5, 5,  5,  5]
        movement_color = "grey"
        for p in range(len(box_x_mins)):
            ax1.add_patch(patches.Rectangle( (box_x_mins[p], ymin1),
                                             box_w_widths[p], height1,
                                             facecolor=movement_color,
                                             edgecolor='none',
                                             alpha=0.2
            ))
            ax2.add_patch(patches.Rectangle( (box_x_mins[p], ymin2),
                                             box_w_widths[p], height2,
                                             facecolor=movement_color,
                                             edgecolor='none',
                                             alpha=0.2
            ))


        ax1.legend(["wall power", "battery power", "accelerometer active", "muscle tensed"],
                  # bbox_to_anchor=(.32, .25),
                  # bbox_transform=plt.gcf().transFigure,
                  # frameon=False
        )


    plt.show()

def plot_eye_alpha():
    # for 1/5/17 recording test
    # data_directory = "/home/em/prog/linux-64-master/2017-01-05_00-15-09/"
    data_directory = "/home/em/prog/linux-64-master/2017-01-05_00-19-20/" # with accel, but looks better for alpha
    figure_directory = "fig_eye_alpha"
    all_eeg_channels = range(9,33)
    # plot_eeg_channels = [12]
    plot_eeg_channels = [32]
    # plot_eeg_channels = all_eeg_channels
    eeg_downsample_factor = 75
    eeg_lowpass_cutoff = 70
    eyes_closed_intervals = [(5,10), (15,20), (25,30), (35,40), (45,50), (55,60)]
    # subtitle = "Alpha power increases during the 5 second periods when eyes are closed."
    subtitle = "Suspected EMG artifacts during the 5 second periods when eyes are closed."
    # subtitle = "Eyes open/close every 5 seconds"
    use_CAR = True

    Fs_openephys = 30000  # this is a constant (unless we change openephys settings)
    Fs_eeg = Fs_openephys # this should change when the eeg data is downsampled

    recording_number = 1
    data_session_name = os.path.basename(data_directory.strip('/')) + (' rec%d' % recording_number)
    print(data_session_name)
    os.chdir(data_directory)

    ##### load eeg data
    (x_eeg_all, t_eeg) = load_eeg(data_directory, Fs_openephys, all_eeg_channels, recording_number)

    ##### make figure directory
    make_directory(figure_directory)

    ### filter, reference, and downsample eeg
    (x_eeg_all, t_eeg, Fs_eeg) = preprocess_eeg(x_eeg_all, t_eeg, eeg_lowpass_cutoff, eeg_downsample_factor, Fs_openephys, use_CAR=use_CAR)

    fig = plt.figure()
    ax = fig.gca()

    for chan_name in plot_eeg_channels:
        chan_index = all_eeg_channels.index(chan_name)

        # commit = "2e44da5"
        commit = "____"
        title_str = ('%s, Fs=%d, Fc_low=%d, chan %d, commit %s\n%s' % (data_session_name, Fs_eeg, eeg_lowpass_cutoff, chan_name, commit, subtitle))
        calc_and_plot_spectrogram(ax,
                                  x_eeg_all[chan_index],
                                  t_eeg,
                                  Fs_eeg,
                                  freq_range=[0, eeg_lowpass_cutoff],
                                  title=title_str)

        ymin = -2
        box_height = 1
        for interval in eyes_closed_intervals:
            ax.add_patch(patches.Rectangle((interval[0], ymin),
                                           interval[1] - interval[0], box_height,
                                           facecolor='grey',
                                           edgecolor='none',
                                           alpha=1.0
            ))

        # ax.relim()
        ax.autoscale_view(True, True, True)
        ax.legend(["Eyes closed \n(approximate times)"])

        fig.savefig(figure_directory + '/chan_%02d_eye_freq_star.png' % chan_name)
        ax.cla()

def plot_sync_stuff():
    # data_directory = "/home/em/prog/linux-64-master/2017-01-05_00-37-06/"
    recording_number = 1
    data_session_name = os.path.basename(data_directory.strip('/')) + (' rec%d' % recording_number)
    print(data_session_name)
    os.chdir(data_directory)

    Fs_openephys = 30000  # this is a constant (unless we change openephys settings)
    all_adc_channels = range(1,9)

    (x_adc_all, t_adc) = load_analog_in(data_directory, Fs_openephys, all_adc_channels, recording_number)

def main():

    # filename_chunk_pin1 =  "100_ADC6_2.continuous"
    # filename_chunk_pin2 =  "100_ADC7_2.continuous"
    # filename_motion =      "motion9-27-16_2.txt"
    # data_directory = "/home/em/prog/linux-64-master/2017-01-05_00-15-09/"
    # data_directory = "/home/em/prog/linux-64-master/2017-01-05_00-37-06/"
    # data_directory = "/home/em/data/eeg_tests/2017-01-30_19-17-10"
    data_directory = "/home/em/data/eeg_tests/2017-01-30_18-51-25"
    recording_number = 1
    data_session_name = os.path.basename(data_directory.strip('/')) + (' rec%d' % recording_number)
    print(data_session_name)
    os.chdir(data_directory)

    Fs_openephys = 30000  # this is a constant (unless we change openephys settings)
    Fs_eeg = Fs_openephys # this should change when the eeg data is downsampled

    # eeg_downsample_factor = 30
    eeg_downsample_factor = 75
    eeg_lowpass_cutoff = 70

    # all_eeg_channels = range(1,3)
    # all_eeg_channels = [2, 4, 6, 11, 12, 15, 24]
    all_eeg_channels = range(1,33)
    # all_eeg_channels = [1, 8, 24]
    # all_eeg_channels = range(9,33)

    ##### load eeg data
    (x_eeg_all, t_eeg) = load_eeg(data_directory, Fs_openephys, all_eeg_channels, recording_number)
    print(x_eeg_all)
    print(x_eeg_all.shape)
    exit(3)


    ##### make figure directory
    figure_directory = "fig_quick_summaries"
    make_directory(figure_directory)

    plot_quick_summary(all_eeg_channels, x_eeg_all, t_eeg, Fs_eeg, Fs_openephys,
                       eeg_downsample_factor, eeg_lowpass_cutoff,
                       data_session_name, figure_directory, use_CAR=use_CAR)



main()
# plot_eye_alpha()
# plot_sync_stuff()

class AnalogData:

    def __init__(self, x_all, t, original_Fs, channel_nums):
        # 2d numpy array, len(channel_nums) by len(t)
        self.x_all = x_all

        # 1d numpy array, timepoints for each sample
        self.t = t

        # Original sampling rate (constant determined by open ephys settings)
        self.original_Fs = original_Fs

        # Current sampling rate (may change if downsampled)
        self.Fs = original_Fs

        # List of channel numbers that match the numbers in open ephys file
        #  names. These are NOT indices into x_all.
        self.channel_nums = channel_nums


class Session:
    # TODO add motion data class

    # Assume that every recording is in a separate directory, so we don't have
    # to keep track of the recording number appended to the end of the open
    # ephys filenames. This requires that you restart the open ephys GUI
    # between each time you press record.

    # this sample rate is a constant (unless we change open ephys settings)
    Fs_openephys = 30000
    open_ephys_wrapper = OpenEphysWrapper()

    def __init__(self, directory, name=None, eeg_data=None, sync_data=None):
        self.directory = directory # path of directory containing data files

        if name is not None:
            self.name = name  # the name of the session, for use in figure titles etc
        else:
            # if a session name is not supplied, use the name of the data directory
            self.name = os.path.basename(directory.strip('/'))

        if eeg_data is not None:
            self.eeg_data = eeg_data    # AnalogData object containing EEG data
        if sync_data is not None:
            self.sync_data = sync_data  # AnalogData object containing sync pulses

    def load_eeg(self, channel_nums):
        if self.eeg_data is not None:
            raise RuntimeError("EEG data has already been loaded, what are you doing?")

        (x_all, t) = open_ephys_wrapper.load_continuous_files('100_CH', self.directory, self.Fs_openephys, channel_nums)
        self.eeg_data = AnalogData(x_all, t, self.Fs_openephys, channel_nums)

    # def load_analog_in(data_directory, Fs_openephys, all_channels, recording_number=1):
    #     return load_openephys_files('100_ADC',data_directory, Fs_openephys, all_channels, recording_number)

class OpenEphysWrapper:
    def __init__():
        # I dunno

    def load_continuous_file(path, Fs_openephys):
        # always constant for openephys format, at least as of now
        SAMPLES_PER_RECORD = 1024
        all = ep.loadContinuous(path)
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

    def load_continuous_files(prefix, data_directory, Fs_openephys, channel_names, recording_number=1):
        x_all = []
        last_t = None
        t = None
        # multiple recordings on the same day have _2, _3, ... in the file name
        if recording_number == 1:
            recording_number_str = ''
        elif recording_number > 1:
            recording_number_str = ('_%d' % recording_number)
        else:
            raise RuntimeError('load_openephys_files: invalid recording number')

        for chan_name in channel_names:
            filename = ("%s%d%s.continuous" % (prefix, chan_name, recording_number_str))
            (x, t) = load_openephys_file(data_directory,
                                                filename,
                                                Fs_openephys)
            x_all.append(x)
            if (last_t is not None) and not np.isclose(t, last_t).all():
                raise RuntimeError("load_openephys_files: file timestamps don't match")
            last_t = t

        x_all = np.array(x_all)
        return (x_all, t)
