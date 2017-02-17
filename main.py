#! /bin/python3

import matplotlib.pyplot as plt
from MyAnalysisClasses import Session, AnalogData, Spectrogram



# def plot_quick_summary(all_eeg_channels, x_eeg_all, t_eeg, Fs_eeg, Fs_openephys,
#                        eeg_downsample_factor, eeg_lowpass_cutoff,
#                        data_session_name, figure_directory, use_CAR=True):
#     ##### make figure directory
#     # make_directory(figure_directory)

#     ##### plot all eeg, time and spectrogram
#     fig = plt.figure()
#     ax = fig.gca()

#     ### plot raw eeg
#     for chan_name in all_eeg_channels:
#         title_str = ('%s, Fs=%dHz, channel %d' % (data_session_name, Fs_eeg, chan_name))
#         chan_index = all_eeg_channels.index(chan_name)
#         calc_and_plot_spectrogram(ax,
#                                   x_eeg_all[chan_index],
#                                   t_eeg,
#                                   Fs_eeg,
#                                   title=title_str)
#         fig.savefig(figure_directory + '/chan_%02d_raw.png' % chan_name)
#         ax.cla()

#     ### filter and downsample eeg
#     (x_eeg_all, t_eeg, Fs_eeg) = preprocess_eeg(x_eeg_all, t_eeg, eeg_lowpass_cutoff, eeg_downsample_factor, Fs_openephys, use_CAR=use_CAR)

#     ### plot filtered eeg
#     for chan_name in all_eeg_channels:
#         chan_index = all_eeg_channels.index(chan_name)

#         title_str = ('%s, Fs=%d, lowpass %d, channel %d' % (data_session_name, Fs_eeg, eeg_lowpass_cutoff, chan_name))
#         plot_time(ax, x_eeg_all[chan_index, :], t_eeg,
#                   ylabel='EEG Magnitude',
#                   title=title_str)
#         fig.savefig(figure_directory + '/chan_%02d_time.png' % chan_name)

#         ax.cla()
#         calc_and_plot_spectrogram(ax,
#                                   x_eeg_all[chan_index],
#                                   t_eeg,
#                                   Fs_eeg,
#                                   freq_range=[0, eeg_lowpass_cutoff],
#                                   title=title_str)
#         fig.savefig(figure_directory + '/chan_%02d_freq.png' % chan_name)
#         ax.cla()

def main():
    session = Session("/home/em/data/eeg_tests/2017-01-30_18-51-25")
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
