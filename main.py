#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from MyAnalysisClasses import *

def plot_time_and_freq(session, channel_num, start_time=None, end_time=None, width=None):
    if start_time is None:
        # start from beginning of recording
        start_time = session.eeg_data.t[0]
    if end_time is None:
        # end at end of recording
        end_time = session.eeg_data.t[-1]
    if width is None:
        # plot the whole recording at once, instead of breaking it up into windows
        width = end_time - start_time

    for t in range(start_time, end_time, width):
        fig = plt.figure()
        eeg = session.eeg_data.copy(time_range=[t, t+width]);
        spectrum = Spectrogram(eeg)
        spectrum.calculate_all()
        title = 'Channel %d, %s' % (channel_num, eeg.summary_string())

        # Use the same y-scale for all the time plots.
        # The spectrogram color scaling will change, though...
        props = PlotProperties(title=title, ylim=session.eeg_data.min_max(channel_num))

        axes = plt.subplot(2, 1, 1)
        eeg.plot_channel(channel_num, axes, plot_properties=props)

        axes = plt.subplot(2, 1, 2)
        spectrum.plot_channel(num=channel_num, axes=axes)
        plt.show()


def main():
    session = Session.new('/home/em/prog/plugin-GUI/Builds/Linux/build/2018-06-12_10-43-44')

    channels = range(1,33)
    # channels = [8,9]

    session.load_eeg(channels)
    session.eeg_data.preprocess(downsample_to=400, lowpass_cutoff=70, use_CAR=False)

    for chan in channels:
        plot_time_and_freq(session, chan, 0, 30, 10)



main()
