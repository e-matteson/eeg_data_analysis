#! /bin/python3

import matplotlib.pyplot as plt
import numpy as np
from MyAnalysisClasses import *

def make_fake_data():
    session1 = Session("/home/em/data/eeg_tests/2017-01-30/2017-01-30_18-51-25")
    session1.load_eeg([1,2])
    t = session1.eeg_data.t
    # x_all = np.array([np.sin(t * np.pi) * t * (t<30), np.sin(t * 4 * np.pi) * t * (t>30)])
    x_all = np.array([np.sin(t * 1000* 2*np.pi), np.sin(t * 5000* 2*np.pi)])
    print(t)
    print(x_all)
    data = AnalogData(x_all, t, session1.eeg_data.Fs, [1,2])
    # session = Session("/tmp/notapath", name="testsig", eeg_data=data)
    return data

def test():
    fig = plt.figure()
    axes = fig.gca()
    plot_props = PlotProperties(title='its a plot!', xlabel='im a x-axis')

    data = make_fake_data()
    # new_data = data.copy(time_range=[-1, 300])
    data.plot_channel(1, axes)
    spec = Spectrogram(data)
    spec.calculate_all()
    spec.plot_channel(2, axes)

    # spec.plot_channel(1, axes)
    plt.show()

def main():

    # # # # test()
    # fig = plt.figure()
    # axes = fig.gca()
    # plot_props = PlotProperties(title='its a plot!', xlabel='im a x-axis')
    # t = np.array(range(1,10))
    # x = np.array([t*7, t*8, t*9])
    # # data = AnalogData(x, t, 1)

    # print(x)
    # print(t)
    # TimePlotter.plot_channel(x[0], t, axes, plot_props)
    # plt.show()

    # exit(4)
    # # x = np.array([[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5],
    # #               [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5] ])
    # # print(threshold_01(x, 0.5))
    # # # x = x.transpose()
    # # # t=np.array([1,2,3])
    # # print (x)
    # # print (t)
    # # print(truncate_by_value(x, t, [1.5, 4.1]))

    fig = plt.figure()
    axes = fig.gca()
    plot_props = PlotProperties(title='its a plot!', xlabel='im a x-axis')

    session = Session("/home/em/data/eeg_tests/2017-01-30/2017-01-30_19-17-10")

    # session.load_eeg(range(1,3))

    # foo = session.load_motion('motion-1-30-17.txt', chunk_msb=8, chunk_lsb=7)
    session.load_motion('motion-1-30-17.txt', chunk_msb=8, chunk_lsb=7, enable=6)
    session.motion.plot_sensor(0, axes)
    plt.show()

    exit(5)
    print(session.eeg_data.x_all.shape)
    # data1 = session.eeg_data.copy(index_range=[0, 10*session.eeg_data.Fs])
    data1 = session.eeg_data
    data1.preprocess(downsample_factor=75, lowpass_cutoff=70)
    data1.plot_channel(1, axes)
    spectrum1 = Spectrogram(data1)
    spectrum1.calculate_all()
    spectrum1.plot_channel(1, axes, time_range=[0,10])

    plt.show()
    # data =
    # session.eeg_data.preprocess(downsample_factor=75, lowpass_cutoff=70)
    # session.spectrum = Spectrogram(session.eeg_data)
    # session.spectrum.calculate_all()

    # session.spectrum.plot_channel(1, axes, title="This is a spectrogram!")

    # session.save_fig(fig, "test_figs", "chan_%02d_freq.png" % 1)

main()
