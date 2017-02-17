#! /bin/python3

import matplotlib.pyplot as plt
from MyAnalysisClasses import Session, AnalogData, Spectrogram


def main():
    session = Session("/home/em/data/eeg_tests/2017-01-30/2017-01-30_18-51-25")
    session.load_eeg([1,2])
    session.eeg_data.preprocess(downsample_factor=75, lowpass_cutoff=70)
    print(type(session.eeg_data.x_all[0]))
    session.spectrum = Spectrogram(session.eeg_data)
    session.spectrum.calculate_all()

    fig = plt.figure()
    axes = fig.gca()
    session.spectrum.plot_channel(1, axes, title="This is a spectrogram!")

    session.save_fig(fig, "test_figs", "chan_%02d_freq.png" % 1)

main()
