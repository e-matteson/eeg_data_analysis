#! /bin/python3

import OpenEphys as ope
import numpy as np
import scipy.signal as sig
import scipy.stats as stats

from MyUtilities import *

class OpenEphysWrapper:
    def __init__(self):
        # I dunno
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
        print(channel_nums)
        print(recording_num)
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

    def __init__(self, x_all, t, original_Fs, channel_nums):
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
        self.channel_nums = channel_nums

        # Create empty dictionary to store a record of preprocessing steps
        self.preprocess_config = {}

    def __str__(self):
        string = "<AnalogData: %s, [%0.2f ... %0.2f], %s>" % (self.channel_nums, self.t[0], self.t[-1], self.Fs)
        return string

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


class Session:
    # TODO add motion data class

    # Assume that every recording is in a separate directory, so we don't have
    # to keep track of the recording number appended to the end of the open
    # ephys filenames. This requires that you restart the open ephys GUI
    # between each time you press record.

    # this sample rate is a constant (unless we change open ephys settings)
    Fs_openephys = 30000
    open_ephys = OpenEphysWrapper()

    def __init__(self, directory, name=None, eeg_data=None, sync_data=None):
        self.directory = directory # path of directory containing data files

        if name is not None:
            self.name = name  # the name of the session, for use in figure titles etc
        else:
            # if a session name is not supplied, use the name of the data directory
            self.name = os.path.basename(directory.strip('/'))

        # if eeg_data is not None:
        #     self.eeg_data = eeg_data    # AnalogData object containing EEG data
        # if sync_data is not None:
        #     self.sync_data = sync_data  # AnalogData object containing sync pulses
        self.eeg_data = eeg_data    # AnalogData object containing EEG data
        self.sync_data = sync_data  # AnalogData object containing sync pulses

    def __str__(self):
        string = "<Session: %s, %s, %s>" % (self.name, self.eeg_data, self.sync_data)
        return string

    def load_eeg(self, channel_nums):
        if self.eeg_data is not None:
            raise RuntimeError("EEG data has already been loaded, what are you doing?")

        self.eeg_data = self.open_ephys.load_continuous_channels('100_CH', self.directory, self.Fs_openephys, channel_nums)


    # def load_analog_in(data_directory, Fs_openephys, all_channels, recording_num=1):
    #     return load_continuous_channels('100_ADC',data_directory, Fs_openephys, all_channels, recording_num)

    def save_fig(self, fig, figure_directory, filename):
        """Save the figure object as filename in session_directory/figure_directory"""
        full_dir_path = os.path.join(self.directory, figure_directory)
        make_directory(full_dir_path)
        fig.savefig(os.path.join(full_dir_path, filename))


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
        pxx_all, freq_bins = truncate_to_range(pxx_all, freq_bins, self.config['freq_range'])

        self.pxx_all = pxx_all
        self.freq_bins = freq_bins
        self.time_bins = time_bins

    def plot_channel(self, channel_num, axes, title='',colorbar=None, vmin=None, vmax=None):
        "Plot one channel of data on the given axes. Specify channel by name/number, not index!"
        # TODO implement
        # TODO what do vmin and vmax do? can they be automated?
        channel_index = self.data.channel_num_to_index(channel_num)
        im = axes.imshow(
            # TODO don't transpose anymore! update freq truncation too...
            self.pxx_all[channel_index].transpose(),
            origin="lower",
            aspect="auto",
            extent=[self.time_bins[0], self.time_bins[-1], self.freq_bins[0], self.freq_bins[-1]],
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



    def calculate_1channel(self, data_channel):
        # this could be optimized by computing spectrograms for all channels at once
        # but that would make plotting more complicated
        # TODO what are the actual units of pxx?
        # is the 10*log10 the right way to get dB?

        # resolution = 1024
        (freq_bins, time_bins, pxx) = sig.spectrogram(
            data, fs=Fs,
            nperseg=self.config['resolution'],
            noverlap=int(self.config['resolution']/2),
            mode='psd',
            scaling= 'density')
        # print(time_bins)
        # exit(3)
        pxx = pxx.transpose()
        if self.config['log']:
            pxx = 10*np.log10(pxx/self.config['log_ref'])
        pxx, freq_bins = truncate_to_range(pxx, freq_bins, freq_range)
        return (freq_bins, time_bins, pxx)
