#! /bin/python3

import OpenEphys as ep
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

from MyUtilities import *
from MotionLoading import *

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

def preprocess_eeg(x_eeg_all, t_eeg, eeg_lowpass_cutoff, eeg_downsample_factor, Fs_openephys):
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
    for chan_num in range(x_eeg_all.shape[0]):
        x_eeg_all_new.append(downsample(x_eeg_all[chan_num], factor))

    return (np.array(x_eeg_all_new), t_eeg_new)

def lowpass_all_eeg(x_eeg_all, cutoff, Fs_eeg):
    x_eeg_all_copy = np.zeros(x_eeg_all.shape)
    for chan_num in range(x_eeg_all.shape[0]):
        x_eeg_all_copy[chan_num, :] = lowpass(x_eeg_all[chan_num, :], cutoff, Fs_eeg)
    return x_eeg_all_copy

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


def main():

    filename_chunk_pin1 =  "100_ADC6_2.continuous"
    filename_chunk_pin2 =  "100_ADC7_2.continuous"
    filename_motion =      "motion9-27-16_2.txt"
    folder = "/home/em/new_data/eeg_test_9-27-16/2016-09-27_19-02-40/"
    Fs_openephys = 30000

    eeg_downsample_factor = 20
    eeg_lowpass_cutoff = 100
    all_eeg_channels = range(1,5)
    # all_eeg_channels = range(1,33)

    ##### load eeg data
    (x_eeg_all, t_eeg) = load_all_eeg(folder, Fs_openephys, all_eeg_channels)
    (x_eeg_all, t_eeg, Fs_eeg) = preprocess_eeg(x_eeg_all, t_eeg, eeg_lowpass_cutoff, eeg_downsample_factor, Fs_openephys)

    # # plot_all_eeg_time
    # for chan_num in all_eeg_channels:
    #     fig =

    plt.show()

    ##### load motion data
    (x_motion0, x_motion1, x_motion2, t_motion) =  get_motion(folder,
                                                              filename_motion,
                                                              filename_chunk_pin1,
                                                              filename_chunk_pin2,
                                                              Fs_openephys)
    mvmt_onsets = get_mvmt_onsets()


main()
