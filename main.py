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


def test2():
    # # # test()
    fig = plt.figure()
    axes = fig.gca()
    plot_props = PlotProperties(title='its a plot!', xlabel='im a x-axis')
    t = np.array(range(1,10))
    x = np.array([t*7, t*8, t*9])
    # data = AnalogData(x, t, 1)

    print(x)
    print(t)
    TimePlotter.plot_channel(x[0], t, axes, plot_props)
    plt.show()

    exit(4)
    # x = np.array([[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5],
    #               [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5] ])
    # print(threshold_01(x, 0.5))
    # # x = x.transpose()
    # # t=np.array([1,2,3])
    # print (x)
    # print (t)
    # print(truncate_by_value(x, t, [1.5, 4.1]))



def get_manual_onset_times(motion_data):
    """For session '/home/em/data/eeg_tests/2017-01-30/2017-01-30_19-17-10' """

    times = [63.2, 72.5, 93.2, 103.6, 123.4, 133.5, 153.1, 163.25, 183.15, 193.1,
             213.3, 224.7, 243.7, 253.7, 273.1, 283.9, 303.3, 313.6, 333.6,
             343.8, 363.55, 373.7, 393.5, 407.05, 423.35, 433.3, 453.3, 463.65,
             483.5, 493.5, 513.6, 523.55, 543.65, 554.0, 573.5, 583.65, 603.7,
             613.3]
    return times

def main():

    fig = plt.figure()
    axes = fig.gca()

    session = Session("/home/em/data/eeg_tests/2017-01-30/2017-01-30_19-17-10")
    # maybe 9,10,11,12 are bad?
    # session.load_eeg(list(range(1,9))+list(range(13,33)))
    # session.load_eeg(range(1,33))
    # session.load_eeg(range(1,3))
    # session.eeg_data.preprocess(downsample_factor=75, lowpass_cutoff=70, highpass_cutoff=2, use_CAR=False)
    # session.eeg_data.plot_channel(1, axes)

    session.load_motion('motion-1-30-17.txt', chunk_msb=8, chunk_lsb=7, enable=6)

    for i in range(3):
        subplot_axes  = fig.add_subplot(1,3,i+1)
        sensor = session.motion.sensors[i]
        # subplot_axes.plot(sensor.t, sensor.x_all.transpose())
        session.motion.plot_sensor(i, subplot_axes)
    plt.show()
    exit(3)

    session.spectrum = Spectrogram(session.eeg_data)
    session.spectrum.calculate_all()

    onset_list = get_manual_onset_times(session.motion)
    plot_props = PlotProperties(title='its a plot!', xlabel='Time (s)', ylabel='Mean Amplitude')
    time_interval = [-4, 4]
    fig_dir_name = "fig_onsets_hp"
    # TODO highpass filter is broken! test
    #  and stop remaking filters every time. And decide what filter types to use.
    for channel_num in session.eeg_data.channel_nums:
        title_str = ('%s, Fs=%d, lowpass %0.2f, highpass %0.2f, CAR=%s, channel %d' % (
            session.name, session.spectrum.data.Fs,
            session.eeg_data.preprocess_config['lowpass_cutoff'],
            # session.eeg_data.preprocess_config['highpass_cutoff'],
            session.eeg_data.preprocess_config['use_CAR'],
            channel_num))
        plot_props.title = title_str

        onsets = session.eeg_data.get_intervals(channel_num, onset_list, time_interval)
        # onsets.plot_all(axes)
        lmp = np.mean(onsets.x_all, axis=0)
        TimePlotter.plot_all(lmp, onsets.t, axes, plot_props)
        session.save_fig(fig, fig_dir_name, "chan_%02d_onset_lmp.png" % channel_num)
        plt.cla()

        # TODO is averaging spectrograms like this OK? Especially since their time_bins don't quite line up?
        spec_onsets = session.spectrum.get_intervals(channel_num, onset_list, time_interval)
        spec_onsets.pxx_all = np.array([np.mean(spec_onsets.pxx_all, 0)])
        spec_onsets.plot_channel(index=0, axes=axes, title=title_str,
                                 freq_range=[0, session.eeg_data.preprocess_config['lowpass_cutoff']])
        session.save_fig(fig, fig_dir_name, "chan_%02d_onset_freq.png" % channel_num)
        plt.cla()
        continue
        # print("plotted")
        # plt.show()
        # exit(4)
    print("done")
    exit(4)

    # (x_onsets, t_onsets) = get_peri_onset_intervals(session.spectrum, channel_num, onset_times, time_interval)
    # print(session.eeg_data.x_all.shape)
    # data1 = session.eeg_data.copy(index_range=[0, 10*session.eeg_data.Fs])



    # show_mvmt_onset_lines_over_quats(onset_list, session.motion, axes)

    data1 = session.eeg_data
    interval =  data1.Fs*np.array([-1, 1])

    print(onsets.x_all)
    print(np.mean(onsets.x_all, axis=0))
    axes.plot(onsets.t, np.mean(onsets.x_all, 0))
     #e else = mean()

    exit(3)
    # data1.preprocess(downsample_factor=75, lowpass_cutoff=70)
    # data1.plot_channel(1, axes)
    # spectrum1 = Spectrogram(data1)
    # spectrum1.calculate_all()
    # spectrum1.plot_channel(1, axes, time_range=[0,10])

    plt.show()
    # data =
    # session.eeg_data.preprocess(downsample_factor=75, lowpass_cutoff=70)
    # session.spectrum = Spectrogram(session.eeg_data)
    # session.spectrum.calculate_all()

    # session.spectrum.plot_channel(1, axes, title="This is a spectrogram!")

    # session.save_fig(fig, "test_figs", "chan_%02d_freq.png" % 1)

main()

    # session.motion.plot_sensor(0, axes)
    # for i in range(3):
    #     subplot_axes  = fig.add_subplot(1,3,i+1)
    #     sensor = session.motion.sensors[i]
    #     # subplot_axes.plot(sensor.t, sensor.x_all.transpose())
    #     session.motion.plot_sensor(i, subplot_axes)
    #     # for q in range(4):
    #     #     grad = np.gradient(sensor.x_all[q])
    #     #     print(grad)
    #     #     subplot_axes.plot(sensor.t, grad)
