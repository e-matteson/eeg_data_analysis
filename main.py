#! /bin/python3

import matplotlib.pyplot as plt
import numpy as np
from MyAnalysisClasses import *

def make_fake_data():
    session1 = Session("/home/em/data/eeg_tests/2017-01-30/2017-01-30_18-51-25")
    session1.load_eeg([1,2])
    t = session1.eeg_data.t
    x_all = np.array([np.sin(t * np.pi), np.sin(t * 4 * np.pi)])
    print(t)
    print(x_all)
    data = AnalogData(x_all, t, session1.eeg_data.Fs, [1,2])
    # session = Session("/tmp/notapath", name="testsig", eeg_data=data)
    return data

def main():

    data = make_fake_data()
    fig = plt.figure()
    axes = fig.gca()
    plot_props = PlotProperties(title='its a plot!', xlabel='im a x-axis')
    data.plot_channel(axes, 1)
    plt.show()
    exit(3)


    # session = Session("/home/em/data/eeg_tests/2017-01-30/2017-01-30_19-17-10")

    session.eeg_data.preprocess(downsample_factor=75, lowpass_cutoff=70)
    print(type(session.eeg_data.x_all[0]))
    session.spectrum = Spectrogram(session.eeg_data)
    session.spectrum.calculate_all()

    session.spectrum.plot_channel(1, axes, title="This is a spectrogram!")

    session.save_fig(fig, "test_figs", "chan_%02d_freq.png" % 1)

main()
