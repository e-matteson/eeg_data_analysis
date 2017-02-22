#! /bin/python3

import OpenEphys as ope
import numpy as np
import scipy.signal as sig
import scipy.stats as stats
import copy
import json
import gc

from MyUtilities import *

class OpenEphysWrapper:
    def __init__(self):
        # should this do anying, I dunno
        pass

    def load_continuous(self, path, Fs_openephys):
        # always constant for openephys format, at least as of now
        SAMPLES_PER_RECORD = 1024
        all = ep.loadContinuous(path)
        header = all['header']
        sampleRate = int(header['sampleRate'])
        if sampleRate != Fs_openephys:
            raise RuntimeError('load_continuous: unexpected sample rate')

        x = all['data']

        # get timestamps for each record
        timestamps = all['timestamps']
        # compute timestamps for each sample - I think this is the right way...
        t = np.array([np.arange(time, time+1024) for time in timestamps]).flatten() / sampleRate
        if not are_intervals_close(t, 1/sampleRate):
            raise RuntimeError('load_continuous: timestamps may be wrong')
        return (x,t)

    def load_continuous_channels(self, prefix, data_directory, Fs_openephys, channel_nums, recording_num=1):
        # Return an AnalogData object containing the data from each continuous
        #  file in the given directory with the given prefix and channel number.
        x_all = []
        last_t = None
        t = None
        # multiple recordings on the same day have _2, _3, ... in the file name
        if recording_num == 1:
            recording_number_str = ''
        elif recording_num > 1:
            recording_number_str = ('_%d' % recording_num)
        else:
            raise RuntimeError('load_continuous_channels: invalid recording number')

        for chan_name in channel_nums:
            filename = ("%s%d%s.continuous" % (prefix, chan_name, recording_number_str))
            (x, t) = self.load_continuous(os.path.join(data_directory, filename),
                                                Fs_openephys)
            x_all.append(x)
            if (last_t is not None) and not np.isclose(t, last_t).all():
                raise RuntimeError("load_continuous_channels: file timestamps don't match")
            last_t = t

        # TODO uh oh, is this a 1d np.array of lists? or a 2d np.array? Seems right, 2d array.
        x_all = np.array(x_all)
        data = AnalogData(x_all, t, Fs_openephys, channel_nums)

        return data



class AnalogData:

    def __init__(self, x_all, t, original_Fs, channel_nums=None):
        # 2d numpy array, len(channel_nums) by len(t)
        self.x_all = x_all

        # 1d numpy array, timepoints for each sample
        self.t = t

        # TODO is Fs versus original_Fs handled correctly everywhere?
        # Original sampling rate (constant determined by open ephys settings)
        self.original_Fs = original_Fs

        # Current sampling rate (may change if downsampled)
        self.Fs = original_Fs

        # List of channel numbers that match the numbers in open ephys file
        #  names. These are NOT indices into x_all.
        if channel_nums is None:
            self.channel_nums = range(x_all.shape[0])
        else:
            self.channel_nums = channel_nums

        # Create empty dictionary to store a record of preprocessing steps
        self.preprocess_config = {}

    def __str__(self):
        string = "<AnalogData: %s, [%0.2f ... %0.2f], %s>" % (self.channel_nums, self.t[0], self.t[-1], self.Fs)
        return string

    def copy(self, time_range=None, index_range=None):
        """Return a new analogData object with a copy of everything in this one. Optionally truncate the data of the new copy."""
        # TODO There's a whole lot of inefficient copying going on here.
        # Ideas: - make truncate functions modify data instead of copying it
        #        - only copy the requested range from the old AnalogData.
        new = copy.deepcopy(self)

        if time_range is not None and index_range is not None:
            raise RuntimeError('You may not supply both time_range and index_range')

        if time_range is not None:
            new.truncate_value(time_range)
        elif index_range is not None:
            new.truncate_index(index_range)
        return new

    def length(self):
        assert(self.t.shape[0] == self.x_all.shape[1])
        return self.t.shape[0]

    def count_channels(self):
        """Return the number of channels of data"""
        count = len(self.channel_nums)
        if count != self.x_all.shape[0]:
            raise RuntimeError("inconsistent channel count")
        return count

    def get_channel(self, number):
        """Get 1d numpy array channel by number/name (not by index!)"""
        return self.x_all[self.channel_num_to_index(number)]

    def channel_num_to_index(self, number):
        """Get index into x_all that corresponds to the given channel number/name"""
        try:
            channel_index = self.channel_nums.index(number)
        except ValueError:
            raise RuntimeError("channel number %d does not exist" % number)
        return channel_index

    def preprocess(self, downsample_factor=None, lowpass_cutoff=None, use_CAR=True):
        # TODO support other types of filtering

        # Store settings, for future reference
        self.preprocess_config['downsample_factor'] = downsample_factor
        self.preprocess_config['lowpass_cutoff'] = lowpass_cutoff
        self.preprocess_config['use_CAR'] = use_CAR

        # common average reference
        if use_CAR:
            self.common_avg_reference()

        # lowpass filter
        if lowpass_cutoff is not None:
            self.lowpass(lowpass_cutoff)

        # downsample to a new sample rate
        if downsample_factor is not None:
            new_Fs = self.Fs / downsample_factor
            if new_Fs != int(new_Fs):
                raise RuntimeError("Can't downsample to non-integer sample rate: Fs = %f Hz" % new_Fs)
            if (lowpass_cutoff is None) or (new_Fs < lowpass_cutoff*2):
                raise RuntimeWarning("Aliasing may occur! cutoff = %f Hz, Fs = %f Hz" % (lowpass_cutoff, new_Fs))
            print("Downsampling to %0.2f Hz" % new_Fs)
            self.downsample(downsample_factor)
            self.Fs = new_Fs


    def common_avg_reference(self):
        """Perform common average referencing on the data channels."""
        common_avg = np.mean(self.x_all, 0)
        self.x_all = self.x_all - common_avg

    def downsample(self, factor):
        """Downsample all the data channels.
        Does not include an anti-aliasing filter!!!"""
        self.t = downsample(self.t, factor)
        print()
        new_x_all = []
        for chan_index in range(self.count_channels()):
            # self.x_all[chan_index, :] = downsample(self.x_all[chan_index, :], factor)
            new_x_all.append(downsample(self.x_all[chan_index, :], factor))
        self.x_all = np.array(new_x_all)

    def lowpass(self, cutoff):
        """Lowpass filter all data channels to the given cutoff frequency (Hz)"""
        # x_eeg_all_copy = np.zeros(x_eeg_all.shape)
        for chan_index in range(self.count_channels()):
            self.x_all[chan_index, :] = lowpass(self.x_all[chan_index, :], cutoff, self.Fs)

    def plot_channel(self, channel_num, axes, plot_properties=None):
        if plot_properties is None:
            plot_properties = PlotProperties(xlabel="Time (s)", ylabel="Amplitude")
        channel_index = self.channel_num_to_index(channel_num)
        TimePlotter.plot_channel(self.x_all[channel_index], self.t, axes, plot_properties)

    def plot_all(self, axes, plot_properties=None):
        if plot_properties is None:
            plot_properties = PlotProperties(xlabel="Time (s)", ylabel="Amplitude")
        TimePlotter.plot_all(self.x_all, self.t, axes, plot_properties)

    def truncate_value(self, time_range):
        (self.x_all, self.t) = truncate_by_value(self.x_all, self.t, time_range)

    def truncate_index(self, index_range):
        (self.x_all, self.t) = truncate_by_index(self.x_all, self.t, index_range)

    def get_intervals(self, channel_num, onset_times, interval_times):
        """Return """
        x_onsets = []
        t_onset = None
        channel_index = self.channel_num_to_index(channel_num)
        interval_times = np.array(interval_times)
        for onset in onset_times:
            time_range = interval_times + onset
            (x_interval, t_interval) = truncate_by_value(self.x_all[channel_index], self.t, time_range)
            gc.collect() # force garbage collection, or we'll run out of memory
            x_onsets.append(x_interval)
            new_t_onset = t_interval - onset
            if t_onset is None:
                t_onset = new_t_onset
            elif not np.isclose(t_onset, new_t_onset).all():
                raise RuntimeError("onset time arrays are inconsistent")
        x_onsets = np.array(x_onsets)
        t_onset = np.array(t_onset)
        onsets = AnalogData(x_onsets, t_onset, self.Fs)
        print(x_onsets)
        print(t_onset)
        return onsets

        # return x_onsets
        # x_mean_onset = np.mean(x_onsets, axis=0)
        # x_sem_onset = stats.sem(x_onsets, axis=0)
        # # x_std_onset = np.std(x_onsets, axis=0)
        # t_mean_onset = (np.arange(len(x_mean_onset)) - num_samples_before) / Fs_eeg # off by one?
        # return (x_mean_onset, x_sem_onset, t_mean_onset, x_onsets)


class PlotProperties:
    def __init__(self, title='', xlabel='', ylabel='', linestyle=None, marker=None,
                 linewidth=1, xlim=None, ylim=None, color=None):
        # TODO add legend, especially for MotionData
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.linewidth =linewidth
        self.linestyle = linestyle
        self.marker = marker
        self.xlim = xlim
        self.ylim = ylim
        self.color = color

class Session:
    # Assume that every recording is in a separate directory, so we don't have
    # to keep track of the recording number appended to the end of the open
    # ephys filenames. This requires that you restart the open ephys GUI
    # between each time you press record.

    # this sample rate is a constant (unless we change open ephys settings)
    Fs_openephys = 30000
    open_ephys = OpenEphysWrapper()

    def __init__(self, directory, name=None, eeg_data=None):
        self.directory = directory # path of directory containing data files

        if name is not None:
            self.name = name  # the name of the session, for use in figure titles etc
        else:
            # if a session name is not supplied, use the name of the data directory
            self.name = os.path.basename(directory.strip('/'))

        self.eeg_data = eeg_data    # AnalogData object containing EEG data
        self.motion = None          # MotionData object

    def __str__(self):
        string = "<Session: %s, %s, %s>" % (self.name, self.eeg_data, self.sync_data)
        return string

    def load_eeg(self, channel_nums):
        if self.eeg_data is not None:
            raise RuntimeError("EEG data has already been loaded, what are you doing?")

        self.eeg_data = self.open_ephys.load_continuous_channels('100_CH', self.directory, self.Fs_openephys, channel_nums)


    def load_motion(self, motion_file, chunk_msb=None, chunk_lsb=None, enable=None):
        # TODO organize this better
        chunk_pins = self.open_ephys.load_continuous_channels('100_ADC', self.directory, self.Fs_openephys, [chunk_msb, chunk_lsb])
        enable_pin = self.open_ephys.load_continuous_channels('100_ADC', self.directory, self.Fs_openephys, [enable])
        enable_index = MotionLoader.find_enable_index(enable_pin)
        print("enable time: %f" % chunk_pins.t[enable_index])

        chunk_nums_array = MotionLoader.get_chunk_nums(msb=chunk_pins.x_all[0], lsb=chunk_pins.x_all[1])
        (x_motion, samples_per_chunk) = MotionLoader.load_motion_file(self.directory, motion_file, num_sensors=3)

        t_motion = MotionLoader.make_motion_timestamps(chunk_nums_array, chunk_pins.t, enable_index, samples_per_chunk)


        sensors = []
        for i in range(x_motion.shape[0]):
            # TODO stop cutting off extra samples to make x_motion and t_motion the same size.
            # Figure out how to make timestamps correctly instead!!!!
            x = x_motion[i, :, :t_motion.shape[0]]

            x = MotionLoader.unwrap_quat(x)
            sensors.append(AnalogData(x, t_motion, self.Fs_openephys))

        self.motion = MotionData(sensors=sensors)


    def save_fig(self, fig, figure_directory, filename):
        """Save the figure object as filename in session_directory/figure_directory"""
        full_dir_path = os.path.join(self.directory, figure_directory)
        make_directory(full_dir_path)
        fig.savefig(os.path.join(full_dir_path, filename))

class TimePlotter:
    # def __init__(self):

    def plot_channel(x, t, axes, props):
        """Plot one channel of data on the given axes. Specify channel by
        name/number, not index! props is a PlotProperties object."""
        # TODO is this the best way to re-use plot_all?
        TimePlotter.plot_all(np.array([x]), t, axes, props)

    def plot_all(x_all, t, axes, props):
        """Plot all channel of data on the given axes. Specify channel by
        name/number, not index! props is a PlotProperties object."""
        # TODO add legend
        # TODO setting somethings twice?
        print(x_all.shape)
        print(t.shape)
        assert(t.shape[0] == x_all.shape[-1])
        axes.plot(t, x_all.transpose(), linestyle=props.linestyle, marker=props.marker,
                  linewidth=props.linewidth, color=props.color)

        if props.title is not None:
            axes.set_title(props.title)
        if props.xlabel is not None:
            axes.set_xlabel(props.xlabel)
        if props.ylabel is not None:
            axes.set_ylabel(props.ylabel)

        if props.marker is not None:
            axes.set_marker(props.marker)
        if props.ylim is not None:
            axes.set_ylim(props.ylim)
        if not props.xlim:
            axes.set_xlim([np.min(t), np.max(t)])
        else:
            axes.set_xlim(props.xlim)

class Spectrogram:

    def __init__(self, analogData, resolution=256, freq_range=None, log=True, log_ref=1):
        self.config = {}
        self.config['resolution'] = resolution
        self.config['freq_range'] = freq_range
        self.config['log'] = log
        self.config['log_ref'] = log_ref
        self.data = analogData

        self.xlabel='Time (s)'
        self.ylabel='Frequency (Hz)'
        if log:
            self.zlabel='PSD (dB)'
        else:
            self.zlabel='PSD'


    def calculate_all(self):
        # TODO check if this is all right, after switching from 1 channel to all channels.
        (freq_bins, time_bins, pxx_all) = sig.spectrogram(
            self.data.x_all, fs=self.data.Fs,
            nperseg=self.config['resolution'],
            noverlap=int(self.config['resolution']/2),
            mode='psd',
            scaling= 'density',
            axis=-1)
        # print(time_bins)
        # exit(3)
        print(time_bins.shape)
        print(pxx_all.shape)
        pxx_all = pxx_all.transpose((0,2,1))
        print(pxx_all.shape)
        if self.config['log']:
            pxx_all = 10*np.log10(pxx_all/self.config['log_ref'])
        pxx_all, freq_bins = truncate_by_value(pxx_all, freq_bins, self.config['freq_range'])
        print(time_bins.shape)
        print(pxx_all.shape)

        exit(2)
        self.pxx_all = pxx_all
        self.freq_bins = freq_bins
        self.time_bins = time_bins

    def plot_channel(self, channel_num, axes, title='', colorbar=None, vmin=None, vmax=None, time_range=None):
        "Plot the spectrogram of one channel on the given axes. Specify channel by name/number, not index!"
        # TODO implement
        # TODO what do vmin and vmax do? can they be automated?
        # TODO use PlotProperties
        channel_index = self.data.channel_num_to_index(channel_num)

        pxx = self.pxx_all[channel_index]
        print(self.pxx_all.shape)
        print(pxx.shape)
        exit(3)
        time_bins = self.time_bins
        if time_range is not None:
            (pxx, time_bins) = truncate_by_value(pxx, time_bins, time_range)

        im = axes.imshow(
            # TODO don't transpose anymore? update freq truncation too...
            pxx.transpose(),
            origin="lower",
            aspect="auto",
            extent=[time_bins[0], time_bins[-1], freq_bins[0], self.freq_bins[-1]],
            cmap=plt.cm.gist_heat,
            interpolation="none",
            vmin=vmin,
            vmax=vmax)

        # cmap=plt.cm.gist_heat, interpolation="hanning")
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_title(title)
        # print (title)
        # if clim is not None:
        #     axes.clim(clim)
        if colorbar is not None:
            plt.colorbar(im, ax=axes, orientation='horizontal').set_label(self.zlabel)

    # def calculate_channel(self, data_channel):
    #     # this could be optimized by computing spectrograms for all channels at once
    #     # but that would make plotting more complicated
    #     # TODO what are the actual units of pxx?
    #     # is the 10*log10 the right way to get dB?

    #     # resolution = 1024
    #     # data =
    #     # channel_index = self.data.channel_num_to_index(channel_num)
    #     (freq_bins, time_bins, pxx) = sig.spectrogram(
    #         data, fs=Fs,
    #         nperseg=self.config['resolution'],
    #         noverlap=int(self.config['resolution']/2),
    #         mode='psd',
    #         scaling= 'density')
    #     # print(time_bins)
    #     # exit(3)
    #     pxx = pxx.transpose()
    #     if self.config['log']:
    #         pxx = 10*np.log10(pxx/self.config['log_ref'])
    #     pxx, freq_bins = truncate_by_value(pxx, freq_bins, freq_range)
    #     return (freq_bins, time_bins, pxx)

class MotionData:
    """A class for storing motion data recorded by the beaglebone / bno055 system"""
    def __init__(self, sensors=None):
         """sensors is a list of AnalogData objects containing motion samples from each sensor"""
         # TODO consider storing sensors in dictionary
         # TODO consider storing other things, like enable index
         self.sensors=sensors

    def num_sensors():
        return len(self.sensors)

    # def preprocess():
    #     for i in range(num_sensors):
    #         self.sensors[i] =
    #         # subplot_axes  = fig.add_subplot(1,3,i+1)
    #         sensor = session.motion.sensors[i]
    #         # subplot_axes.plot(sensor.t, sensor.x_all.transpose())
    #         session.motion.plot_sensor(i, subplot_axes)

    def plot_sensor(self, sensor_index, axes, plot_properties=None):
        x_all = self.sensors[sensor_index].x_all
        t = self.sensors[sensor_index].t
        if plot_properties is None:
            plot_properties = PlotProperties(xlabel="Time (s)", ylabel="Orientation (quaternion)",
                                             title="Motion Sensor %d" % sensor_index)
        print("plotting sensor...")
        TimePlotter.plot_all(x_all, t, axes, plot_properties)



class MotionLoader:

    def get_chunk_nums(msb=None, lsb=None):
        # "static" function
        # chunk_pins is an AnalogData object. MSB pin must be first, LSB pin must be second.
        voltage_threshold = 2.0
        min_samples_per_chunk = 10
        # check dimensions

        assert(len(msb.shape) == 1)
        assert(len(lsb.shape) == 1)
        assert(lsb.size == msb.size )

        # threshold the analog recordings
        msb = threshold_01(msb, voltage_threshold)
        lsb = threshold_01(lsb, voltage_threshold)

        # Calculate the chunk nums encoded by the pins
        chunk_nums = 2*msb + lsb

        # Debounce, because the 2 chunk pins don't change simultaneously, and
        #  sometimes the sample falls between the 2 change times.
        chunk_nums = debounce_discrete_signal(chunk_nums, min_samples_per_chunk)
        return chunk_nums

    def load_motion_file(directory, filename, num_sensors=None):
        """Return 3 4xN arrays, with quaternion data for each sensor."""
        sample_dict_list = []
        with open(os.path.join(directory, filename), 'r') as f:
            for line in f:
                if line.strip()[0] != '#': # not a comment
                    sample_dict_list.append(json.loads(line))

        if num_sensors is not None:
            if len(sample_dict_list[0]['data']) != num_sensors:
                raise RuntimeError("Expected %d motion sensors" % num_sensors)
        else:
            num_sensors = len(sample_dict_list[0]['data'])

        # find samples per chunk, by checking the sample num before a zero
        for i in range(1, len(sample_dict_list)):
            if int(sample_dict_list[i]['sample']) == 0:
                samples_per_chunk = 1+int(sample_dict_list[i-1]['sample'])
                break

        # print(sample_dict_list[0])
        x_motion = []
        for s in range(num_sensors):
            x_motion.append(MotionLoader.get_motion_values(sample_dict_list, s))
        x_motion = np.array(x_motion)
        # print(x_motion)
        return (x_motion, samples_per_chunk)
        # return x_motion

    def get_motion_values(sample_dict_list, sensor_num):
        """Extract quaternion data for 1 sensor, converting hex strings to ints.
        Return 4xN array."""
        # get hex string data for sensor
        x = [d['data'][sensor_num] for d in sample_dict_list]
        # convert hex string to int
        x = [[int(hex_val, 16) for hex_val in sample] for sample in x]
        # convert to numpy array, reshape so quaternion arrays are in first dimension
        x = np.transpose(np.array(x))
        return x

    def make_motion_timestamps(chunk_nums, t_chunk, enable_index, samples_per_chunk):
        """The motion sample rate is irregular! Return irregular timestamps.
        Assume samples are evenly spaced within each chunk."""
        # TODO rewrite docstring
        # TODO will this always behave at the end of a file?
        # TODO is this the best way to extract the timestamps?

        # find the start of each chunk
        chunk_start_indices = []
        for index in range(enable_index, len(t_chunk)):
            # TODO off-by-1?
            if chunk_nums[index] != chunk_nums[index-1]:
                chunk_start_indices.append(index)

        # make evenly spaced sample timestamps within each chunk
        sample_indices = []
        for c in range(len(chunk_start_indices)-1):
            start = chunk_start_indices[c]
            end = chunk_start_indices[c+1]
            interval = (end - start)*1.0/samples_per_chunk
            sample_indices.append(np.arange(start, end, interval))

        # convert from indices to seconds
        sample_indices = np.array(sample_indices, dtype=np.int32).flatten()
        t_motion = t_chunk[sample_indices]

        # if len(t_motion) != x_motion.shape[1]:
        #     raise RuntimeError('Something is very wrong')
        return t_motion


    def find_enable_index(enable_pin):
        """Take enable pin recording as AnalogData object, return the index of the first time it goes high"""
        # TODO check if it is enabled and disabled multiple times
        x = enable_pin.x_all[0]
        x = threshold_01(x, 0.5)
        # print(x)
        for index in range(1, x.shape[0]):
            if x[index-1] == 0 and x[index] == 1:
                # success
                print ("found enable index: %d" % index)
                return index
        # fail
        raise RuntimeError("Failed to find 'enable index'")

    def unwrap_quat(x_motion, range_size=2**16):
        """ Remove discontinuities from quaternion data, by letting values go above and below the range."""
        # TODO I don't know how quaternions are supposed to work, is this valid?
        max_jump_size = range_size/2.0
        x_motion_copy = x_motion.copy()
        for d in range(x_motion_copy.shape[0]): # for w,x,y,z
            for i in range(1, x_motion_copy.shape[1]): # for each sample
                jump = (x_motion_copy[d][i] - x_motion_copy[d][i-1])
                if jump > max_jump_size:
                    # huge jump up, shift back down
                    x_motion_copy[d][i] -= range_size
                elif jump < -max_jump_size:
                    # huge jump down, shift back up
                    x_motion_copy[d][i] += range_size
        return x_motion_copy

# def get_peri_onset_intervals(t, onset_times, time_interval):
#     # TODO take onset times (seconds) instead of indices?
#     # index interval is relative to onset at 0 (eg. [-100, 500])
#     # num_samples_before =
#     # num_samples_after  = time_interval[1] * analog_data.Fs
#     index_intervals = []
#     for onset_time in onset_times:
#         # if ons
#         if
#         onset_index = np.searchsorted(analog_data[], onset)
#         start = onset_index+index_interval[0]
#         end = onset_index+index_interval[1]
#         index_intervals.append([start,end])
#         # x_onsets.append(analog_data.x_all[channel_index, start:end])
#     return index_intervals


def plot_mvmt_onset_lines(ax, onset_indices, line_y_coords, Fs_eeg, color='k'):
    for onset in onset_indices:
        onset_time = onset / Fs_eeg
        print(onset)
        print(onset_time)
        ax.plot([onset_time, onset_time+1.0/Fs_eeg], line_y_coords, color=color, linestyle='--')

def show_mvmt_onset_lines_over_quats(onset_indices, motion, axes):

    Fs = motion.sensors[0].Fs
    motion.plot_sensor(0, axes)
    plot_mvmt_onset_lines(axes, onset_indices, [80000, -20000], Fs)
    # plot_quaternion(ax, x_motion0, t_motion,  xlabel='', ylabel='Hand Orientation ')
    # plot_quaternion(ax, x_motion1, t_motion,  xlabel='', ylabel='Forearm Orientation  ')
    # plot_quaternion(ax, x_motion2, t_motion,  ylabel='Upper Arm Orientation ')
