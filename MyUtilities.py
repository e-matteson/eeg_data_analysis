import OpenEphys as ep
import scipy.io.wavfile
import numpy as np
import scipy.signal as sig
import math
import matplotlib.pyplot as plt

###### LOADING / SAVING UTILITIES
def load_openephys_file(folder, filename, Fs_openephys):
    # always constant for openephys format, at least as of now
    SAMPLES_PER_RECORD = 1024
    all = ep.loadContinuous(folder + filename)
    header = all['header']
    sampleRate = int(header['sampleRate'])
    if sampleRate != Fs_openephys:
        raise RuntimeError('load_openephys_file: unexpected sample rate')

    x = all['data']

    # get timestamps for each record
    timestamps = all['timestamps']
    # compute timestamps for each sample - I think this is the right way...
    t = np.array([np.arange(time, time+1024) for time in timestamps]).flatten() / sampleRate
    if not are_intervals_close(t, 1/sampleRate):
        raise RuntimeError('load_openephys_file: timestamps may be wrong')
    return (x,t)

def save_wav(data, filename, volume=1):
    if volume > 1 or volume < 0:
        raise RuntimeError('save_wav: volume out of range')
    scaled_data = np.int16(data/np.max(np.abs(data)) * volume * 32767)
    scipy.io.wavfile.write(filename, 44100, scaled_data)


###### SIGNAL PROCESSING

def normalize(x):
    return x/x.max()

def threshold_01(x, threshold):
    hi_indices = x > threshold
    x_new = np.zeros(x.shape)
    x_new[hi_indices] = 1
    return x_new

def are_intervals_close(time_array, value):
    return (np.isclose((time_array[1:] - time_array[:-1]), value).all())

def lowpass(data, cutoff, fs, order=5):
    # TODO check dimensions
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype='low', analog=False)
    y = sig.lfilter(b, a, data)
    return y

def downsample(x, factor):
    """Downsample 1D array x by factor.
    Does not include an anti-aliasing filter!!!"""
    x_copy = x.copy()
    pad_size = math.ceil(float(x_copy.size)/factor)*factor - x_copy.size
    x_padded = np.append(x_copy, np.zeros(pad_size)*np.NaN)
    x_new = scipy.nanmean(x_padded.reshape(-1,factor), axis=1)
    return x_new

def unwrap_quat(x_motion, range_size=2**16):
    """ Remove discontinuities from quaternion data, by letting values go above and below the range."""
    # I don't know how quaternions are supposed to work, is this valid?
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

def toDecibels(x, x_ref):
    # x is 1d np array, x_ref is scalar (or same dim array?)
    # TODO is clip() the right solution to zero division?
    # see what options log10 offers
    return 10*np.log10(np.true_divide(x, x_ref).clip(min=1e-8) )
    # return np.true_divide(x, x_ref) ** 10

def debounce_discrete_signal(x, min_samples_per_chunk):
    # Remove any bounces that are shorter than min_samples_per_chunk
    # Do not remove a short leading bounce at the very beginning of the array
    start_index = -1
    x_new = x.copy()
    num_bounces_removed = 0
    for i in range(1, len(x_new)-1):
        # print(start_index)
        if x_new[i] != x_new[i-1]:
            # transition!
            if (start_index > 0) and (i - start_index < min_samples_per_chunk):
                x_new[start_index:i] = x_new[i]
                num_bounces_removed += 1
            else:
                start_index = i
    if num_bounces_removed > 0:
        print('debounce_discrete_signal: removed %d bounces' % num_bounces_removed)
    return x_new


# def quaternion_to_euler(wxyz_list):
# # TODO figure out how to calculate this correctly
#     (w,x,y,z) = tuple(wxyz_list)
#     roll  = math.atan2(2*(w*x + y*z), 1-2*(x**2+ y**2))
#     # print(2*(w*y - z*x))
#     pitch = math.asin(2*(w*y - z*x))

#     yaw   = math.atan2(2*(w*z + x*y), 1-2*(y**2 + z**2))
#     return [roll, pitch, yaw]

def truncate_to_range(x, t, t_range):
    # TODO test!
    # TODO deal with diff x shapes
    # x and t are numpy arrays
    if t_range is None:
        return (x,t)

    new_x = x.copy()
    new_t = t.copy()
    range_indices = [0, t.shape[-1]-1]
    if t_range[0] > t[0]:
        range_indices[0] = np.argmax(t>t_range[0])
    if t_range[1] < t[-1]:
        range_indices[1] = np.argmax(t>t_range[1])
        new_x = new_x[:, range_indices[0]:range_indices[1]]
        new_t = new_t[range_indices[0]:range_indices[1]]
    return (new_x, new_t)

def calc_spectrogram(data, Fs, freq_range=None, log=True, log_ref=1):
    # this could be optimized by computing spectrograms for all channels at once
    # but that would make plotting more complicated
    # TODO what are the actual units of Pxx?
    # is the 10*log10 the right way to get dB?

    # resolution = 1024
    resolution = 256
    (freq_bins, time_bins, Pxx) = sig.spectrogram(
        data, fs=Fs,
        nperseg=resolution,
        noverlap=int(resolution/2),
        mode='psd',
        scaling= 'density')
    # print(time_bins)
    # exit(3)
    Pxx = Pxx.transpose()
    if log:
        Pxx = 10*np.log10(Pxx/log_ref)
    Pxx, freq_bins = truncate_to_range(Pxx, freq_bins, freq_range)
    return (freq_bins, time_bins, Pxx)


####### PLOTTING UTILITIES

def plot_spectrogram(axes, Pxx, time_bins, freq_bins,title='',
                     xlabel='Time (s)',ylabel='Frequency (Hz)',zlabel='PSD (dB)'):
    # TODO WHY ISN'T IT FILLING UP THE TIME AXIS??!!
    print(Pxx.shape)
    print(Pxx.transpose().shape)
    # print(time_bins.shape)
    # print(freq_bins.shape)
    # axes.plot(time_bins, Pxx[:, 128])
    # plt.show()
    # exit(3)
    im = axes.imshow(
        Pxx.transpose(),
        origin="lower",
        aspect="auto",
        extent=[time_bins[0],time_bins[-1],freq_bins[0],freq_bins[-1]],
        cmap=plt.cm.gist_heat,
        interpolation="none")

    # cmap=plt.cm.gist_heat, interpolation="hanning")
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    # plt.colorbar(im, ax=axes, orientation='horizontal').set_label(zlabel)

def calc_and_plot_spectrogram(axes, data, t, Fs, freq_range=None, title='',
                              xlabel='Time (s)',ylabel='Frequency (Hz)',
                              zlabel='PSD (dB)'):

    (freq_bins, time_bins, Pxx)=calc_spectrogram(data, Fs, freq_range=freq_range)
    plot_spectrogram(axes, Pxx, time_bins, freq_bins, title='',
                     xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

def plot_quaternion(axes, x_motion, t_motion, title='', xlabel='Time (s)', ylabel='Amplitude',
                    linestyle='solid', marker=None, linewidth=1, xlim=None, ylim=None, color=None, index_range=[None,None]):
    # Plot each sensor, in the first dimension of x_motion
    for i in range(x_motion.shape[0]):
        plot_time(axes,
                  x_motion[i, index_range[0]:index_range[1]],
                  t_motion[index_range[0]:index_range[1]],
                  title=title, xlabel=xlabel, ylabel=ylabel, linestyle=linestyle,
                  marker=marker, linewidth=linewidth, xlim=xlim, ylim=ylim,
                  color=color)


def plot_time(axes, data, t, title='', xlabel='Time (s)', ylabel='Amplitude',
              linestyle='solid', marker=None, linewidth=1, xlim=None, ylim=None, color=None):

    axes.plot(t[:len(data)], data, linestyle=linestyle, marker=marker, linewidth=linewidth,
              color=color)
    # axes.set_linestyle(linestyle)
    # if marker is not None:
    #     axes.set_marker(marker)
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    if not xlim:
        axes.set_xlim([np.min(t), np.max(t)])
    else:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)

