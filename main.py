#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

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

def plot_mean_onsets(fig, session, time_interval, onset_list, fig_dir_name):
    # Untested, since it was moved into separate function
    axes=fig.gca()
    plot_props = PlotProperties(title='its a plot!', xlabel='Time (s)', ylabel='Mean Amplitude')
    for channel_num in session.eeg_data.channel_nums:
        title_str = ('%s, Fs=%d, lowpass %0.2f, CAR=%s, channel %d' % (
            session.name, session.spectrum.data.Fs,
            session.eeg_data.preprocess_config['lowpass_cutoff'],
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

def plot_all_onsets(fig, session, time_interval, onset_list, fig_dir_name, ylim):
    # TODO set ylim automatically somehow
    axes = fig.gca()

    # ylim = [session.eeg_data.x_all.min(), session.eeg_data.x_all.max()]
    plot_props = PlotProperties(title='its a plot!', xlabel='Time (s)', ylabel='Amplitude', ylim=ylim)
    for channel_num in session.eeg_data.channel_nums:

        onsets = session.eeg_data.get_intervals(channel_num, onset_list, time_interval)
        spec_onsets = session.spectrum.get_intervals(channel_num, onset_list, time_interval)
        assert(len(onsets.channel_nums) == spec_onsets.pxx_all.shape[0])
        for onset_index in onsets.channel_nums:
            # in this case, onset channel names and onset indices are the same, so we kinda mix them together
            print("onset num: ", onset_index)
            # Time Domain:
            title_str = ('%s, Fs=%d, lowpass %0.0f, CAR=%s, channel%d, onset%d' % (
                session.name, session.eeg_data.Fs,
                session.eeg_data.preprocess_config['lowpass_cutoff'],
                session.eeg_data.preprocess_config['use_CAR'],
                channel_num, onset_index))
            plot_props.title = title_str

            onsets.plot_channel(onset_index, axes, plot_props)
            session.save_fig(fig, fig_dir_name, "chan%02d_onset%02d_time.png" % (channel_num, onset_index))
            plt.cla()

            # Frequency Domain:
            spec_onsets.plot_channel(index=onset_index, axes=axes, title=title_str,
                                    freq_range=[0, session.eeg_data.preprocess_config['lowpass_cutoff']])
            session.save_fig(fig, fig_dir_name, "chan%02d_onset%02d_freq.png" % (channel_num, onset_index))
            plt.cla()

def plot_mean_ica_onsets(fig, session, components, ica_spectrum, time_interval, onset_list, fig_dir_name):
    axes=fig.gca()
    plot_props = PlotProperties(title='its a plot!', xlabel='Time (s)', ylabel='Mean Amplitude')
    for channel_num in components.channel_nums:
        title_str = ('%s, Fs=%d, lowpass %0.2f, CAR=%s, ica%d' % (
            session.name, ica_spectrum.data.Fs,
            session.eeg_data.preprocess_config['lowpass_cutoff'],
            session.eeg_data.preprocess_config['use_CAR'],
            channel_num))
        plot_props.title = title_str

        onsets = components.get_intervals(channel_num, onset_list, time_interval)
        # onsets.plot_all(axes)
        lmp = np.mean(onsets.x_all, axis=0)
        TimePlotter.plot_all(lmp, onsets.t, axes, plot_props)
        session.save_fig(fig, fig_dir_name, "ica_%02d_onset_lmp.png" % channel_num)
        plt.cla()

        # TODO is averaging spectrograms like this OK? Especially since their time_bins don't quite line up?
        spec_onsets = ica_spectrum.get_intervals(channel_num, onset_list, time_interval)
        spec_onsets.pxx_all = np.array([np.mean(spec_onsets.pxx_all, 0)])
        spec_onsets.plot_channel(index=0, axes=axes, title=title_str,
                                 freq_range=[0, session.eeg_data.preprocess_config['lowpass_cutoff']])
        session.save_fig(fig, fig_dir_name, "ica_%02d_onset_freq.png" % channel_num)
        plt.cla()

def plot_all_ica_onsets(fig, session, components, ica_spectrum, time_interval, onset_list, fig_dir_name, ylim=None):
    # TODO this is a mess of session and component usage.
    axes = fig.gca()

    # ylim = [session.eeg_data.x_all.min(), session.eeg_data.x_all.max()]
    plot_props = PlotProperties(title='its a plot!', xlabel='Time (s)', ylabel='Amplitude', ylim=ylim)
    for channel_num in components.channel_nums:

        onsets = components.get_intervals(channel_num, onset_list, time_interval)
        spec_onsets = ica_spectrum.get_intervals(channel_num, onset_list, time_interval)
        assert(len(onsets.channel_nums) == spec_onsets.pxx_all.shape[0])
        for onset_index in onsets.channel_nums:
            # in this case, onset channel names and onset indices are the same, so we kinda mix them together
            print("onset num: ", onset_index)
            # Time Domain:
            title_str = ('%s, Fs=%d, lowpass %0.0f, CAR=%s, ICA%d, onset%d' % (
                session.name, components.Fs,
                session.eeg_data.preprocess_config['lowpass_cutoff'],
                session.eeg_data.preprocess_config['use_CAR'],
                channel_num, onset_index))
            plot_props.title = title_str

            onsets.plot_channel(onset_index, axes, plot_props)
            session.save_fig(fig, fig_dir_name, "ica%02d_onset%02d_time.png" % (channel_num, onset_index))
            plt.cla()

            # Frequency Domain:
            spec_onsets.plot_channel(index=onset_index, axes=axes, title=title_str,
                                    freq_range=[0, session.eeg_data.preprocess_config['lowpass_cutoff']])
            session.save_fig(fig, fig_dir_name, "ica%02d_onset%02d_freq.png" % (channel_num, onset_index))
            plt.cla()

def get_manual_onset_times(motion_data):
    """For session '/home/em/data/eeg_tests/2017-01-30/2017-01-30_19-17-10' """

    times = [63.2, 72.5, 93.2, 103.6, 123.4, 133.5, 153.1, 163.25, 183.15, 193.1,
             213.3, 224.7, 243.7, 253.7, 273.1, 283.9, 303.3, 313.6, 333.6,
             343.8, 363.55, 373.7, 393.5, 407.05, 423.35, 433.3, 453.3, 463.65,
             483.5, 493.5, 513.6, 523.55, 543.65, 554.0, 573.5, 583.65, 603.7,
             613.3]
    return times

def plot_all_ica(fig, session, components, ica_spectrum, fig_dir_name):
    axes=fig.gca()
    y_range = [np.amin(components.x_all), np.amax(components.x_all)]
    plot_props = PlotProperties(title='its a plot!', xlabel='time (s)', ylabel='ICA component amplitude')
    for channel_num in components.channel_nums:
        title_str = '%s, Fs=%d, lowpass %0.0f, CAR=%s, ICA %d' % (
                session.name, session.eeg_data.Fs,
                session.eeg_data.preprocess_config['lowpass_cutoff'],
                session.eeg_data.preprocess_config['use_CAR'],
                channel_num)
        plot_props.title = title_str
        plot_props.ylim = y_range

        components.plot_channel(channel_num, axes, plot_props)
        session.save_fig(fig, fig_dir_name, "ica%02d_time.png" % (channel_num))
        plt.cla()

        plot_props.ylim = None
        ica_spectrum.plot_channel(num=channel_num, axes=axes, title=title_str)
        session.save_fig(fig, fig_dir_name, "ica%02d_freq.png" % (channel_num))
        plt.cla()

def plot_motion_sensors(fig, session):
    for i in range(3):
        subplot_axes  = fig.add_subplot(1,3,i+1)
        sensor = session.motion.sensors[i]
        # subplot_axes.plot(sensor.t, sensor.x_all.transpose())
        session.motion.plot_sensor(i, subplot_axes)


def load_data_2017_1_30(from_pickle=False):
    # session = Session('/home/em/data/eeg_tests/2017-01-30/2017-01-30_19-17-10', from_pickle=from_pickle)
    session = Session.new('/home/em/data/eeg_tests/2017-01-30/2017-01-30_19-17-10', from_pickle=from_pickle)

    if from_pickle:
        # we loaded cached data that was already preprocessed, we're done
        return session

    # else, load and preprocess data from scratch
    session.load_motion('motion-1-30-17.txt', chunk_msb=8, chunk_lsb=7, enable=6)

    # maybe 9,10,11,12 are bad?
    session.load_eeg(list(range(1,9))+list(range(13,33)))
    # session.load_eeg([1,2])
    session.eeg_data.preprocess(downsample_factor=75, lowpass_cutoff=70, use_CAR=False)

    session.spectrum = Spectrogram(session.eeg_data)
    session.spectrum.calculate_all()

    session.ica = ica(session.eeg_data, session.eeg_data.count_channels)
    session.ica_spectrum = Spectrogram(session.ica)
    session.ica_spectrum.calculate_all()

    # cache pickled data to a file, for faster loading next time
    session.pickle()

    return session

def get_features(ica_spectrum, ica_nums, freq_indices, num_time_bins, onsets):
    assert(len(ica_spectrum.pxx_all.shape) == 3)
    pxx = ica_spectrum.pxx_all.copy()
    t_pxx = ica_spectrum.time_bins

    # Subtract out the average log power
    avg = np.mean(pxx, 1)
    avg = np.repeat(avg[:, np.newaxis, :], len(t_pxx), axis=1 )
    # print(pxx.shape)
    # print(np.mean(pxx, 1).shape)
    # print(avg.shape)
    pxx = pxx - avg

    # Collect the bins into intervals
    data = []
    t_data = []
    labels = []
    for i in range(len(t_pxx) - num_time_bins):
        sample = []
        for ica_num in ica_nums:
            for freq_index in freq_indices:
                sample.extend(pxx[ica_num,  i:i+num_time_bins, freq_index])
        data.append(sample)
        # TODO what timestamps should go with these bins?
        #  How do the spectrogram time_bins values correspond with the original
        #  timestamps of the samples in the bins?
        t_data.append(t_pxx[i : i + num_time_bins])

        # Set label to 1 if there's a movement onset during the last included bin.
        labels.append(0)
        # print(t_data[-1])
        for onset in onsets:
            if onset > t_data[-1][-2] and onset <= t_data[-1][-1]:
                # print(onset)
                # print(t_data[-1])
                labels[-1] = 1
                break

    data = np.array(data)
    t_data = np.array(t_data)
    labels = np.array(labels)
    return (data, t_data, labels)

def test_forest(data, labels):
    predicted_labels = []
    predicted_probs = []
    for i in range(len(labels)):
        train_data = np.delete(data, i, axis=0)
        train_labels = np.delete(labels, i)
        test_data = np.array([data[i]])
        test_label = labels[i]

        forest = RandomForestClassifier()
        forest.fit(train_data, train_labels)
        # predicted_labels.append(forest.predict(test_data))
        predicted_probs.append(forest.predict_proba(test_data)[0])
        print(test_label, predicted_probs[-1])
        if test_label == 1:
            print(predicted_probs[-1])
        # scores = cross_val_score(forest, data, labels)
    # predicted_labels = np.array(predicted_labels)
    predicted_probs = np.array(predicted_probs)
    # print(predicted_probs)
    # print(labels[:10])
    # print(sum(predicted_labels))

def main_evan():
    # TODO highpass filter is broken! test
    #  and stop remaking filters every time. And decide what filter types to use.

    fig = plt.figure()
    axes = fig.gca()

    # session=load_data_2017_1_30(from_pickle=False)
    session=load_data_2017_1_30(from_pickle=True)

    # plot_motion_sensors(fig, session)
    # plot_all_ica(fig, session, session.ica, session.ica_spectrum, "fig_ica")

    onset_list = get_manual_onset_times(session.motion)
    time_interval = [-4, 4]

    print(session.eeg_data.x_all.shape)
    print(session.ica_spectrum.pxx_all.shape)
    print(session.ica_spectrum.time_bins.shape)
    print(session.ica_spectrum.freq_bins.shape)

    plot_props = PlotProperties(title='its a plot!', xlabel='im a x-axis')

    # for ica_num in range(session.ica_spectrum.pxx_all.shape[0]):
    # ica_num = 31
    freq_index = 7
    (data, t_data, labels) = get_features(session.ica_spectrum, [0,3], [7,10], 6, onset_list)
    print(labels)
    print(np.sum(labels))
    print(labels.shape)
    print(data.shape)
    print(t_data.shape)
    # exit(1)
    test_forest(data, labels)

    exit(3)
    x = []


    #     x = session.ica_spectrum[channel_num]
    #     print(x)
    #     exit(1)
        # spec_onsets = session.ica_spectrum.get_intervals(channel_num, onset_list, time_interval)
        # for onset_index in onsets.channel_nums:
        #     pass

    # plot_mean_onsets(fig, session, time_interval, onset_list, "fig_mean_onsets")
    # plot_all_onsets(fig, session, time_interval, onset_list, "fig_all_onsets", [-60,60])
    # plot_all_ica_onsets(fig, session, session.ica, session.ica_spectrum, time_interval, onset_list, "fig_ica_all_onsets", None)
    # plot_mean_ica_onsets(fig, session, session.ica, session.ica_spectrum, time_interval, onset_list, "fig_ica_mean_onsets")

def main_nathan():
    print("Test movement labeling here.")

def main():
    main_evan()
    # main_nathan()
    # a = np.array([[[0,1,2,3], [4,5,6,7], [8,9,10,11]],
    #               [[12,13,14,15], [16,17,18,19], [20,21,22,23] ]])
    # b = np.mean(a, 1)
    # print(a)
    # print(b)
    # # c = np.tile(b, (1, 3, 1))
    # # c = np.repeat(b, 1, axis=1)
    # # c = np.matlib.repmat(b, (1, 1, 1, 2))
    # c = np.repeat(b[:, np.newaxis, :], 3, axis=1)
    # print(c)
    # print(a.shape)
    # print(b.shape)
    # print(c.shape)

main()
