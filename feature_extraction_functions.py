"""
特征提取函数集合
"""
import numpy
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
from librosa.feature import mfcc
EPS = numpy.finfo(float).eps


def get_feature(fs, signal):
    """
    简易的提取特征函数
    :param fs: 采样率
    :param signal: 信号
    :return feature:mfcc特征
    """
    feature = mfcc(signal, fs, S=None, n_mfcc=20).T
    return feature

def mt_feature_extraction(signal, fs, mt_win, mt_step, st_win):
    """
    提取 mid-term feature和short-term feature
    :param signal: 信号
    :param fs: 采样率
    :param mtWin: mid-term窗口大小
    :param mtStep: mid-term步长
    :param stWin: short-term窗口大小
    :param stStep: short-term步长
    :return: mid-term feature, short-term feature
    """
    st_step = st_win
    mt_win_ratio = int(round(mt_win / st_step))
    mt_step_ratio = int(round(mt_step / st_step))

    st_features = st_feature_extraction(signal, fs, st_win, st_step)
    num_of_features = len(st_features)
    num_of_statistics = 2

    mt_features = []
    for i in range(num_of_statistics * num_of_features):
        mt_features.append([])

    for i in range(num_of_features):
        cur_pos = 0
        len_feat = len(st_features[i])
        while cur_pos < len_feat:
            pos_1 = cur_pos
            pos_2 = cur_pos + mt_win_ratio
            if pos_2 > len_feat:
                pos_2 = len_feat
            cur_st_features = st_features[i][pos_1:pos_2]

            mt_features[i].append(numpy.mean(cur_st_features))
            mt_features[i+num_of_features].append(numpy.std(cur_st_features))
            #mtFeatures[i+2*numOfFeatures].append(numpy.std(cur_st_features) /
            # (numpy.mean(cur_st_features)+EPS))
            cur_pos += mt_step_ratio

    return numpy.array(mt_features), st_features


def st_feature_extraction(signal, fs, win, step):
    """
    提取 short-term feature
    :param signal: 信号
    :param fs: 采样率
    :param win: short-term窗口大小
    :param step: short-term步长
    :return: short-term feature
    """
    win = int(win)
    step = int(step)
    signal = numpy.double(signal)
    signal = signal / (2.0 ** 15)
    dc = signal.mean()
    max_value = (numpy.abs(signal)).max()
    signal = (signal - dc) / (max_value + EPS)

    len_feat = len(signal)
    cur_pos = 0
    count_frames = 0
    n_fft = win / 2

    [fbank, _] = mfcc_init_filter_banks(fs, n_fft)
    n_chroma, n_freqs_per_chroma = st_chroma_features_init(n_fft, fs)

    num_of_time_spectral_features = 8
    num_of_harmonic_features = 0
    nceps = 13
    num_of_chroma_features = 13
    total_num_of_features = num_of_time_spectral_features + nceps + num_of_harmonic_features + \
                         num_of_chroma_features
#    totalNumOfFeatures = num_of_time_spectral_features + nceps + numOfHarmonicFeatures

    st_features = []
    xprev = []
    while cur_pos + win - 1 < len_feat:
        count_frames += 1
        pos_signal = signal[cur_pos:cur_pos+win]
        cur_pos = cur_pos + step
        cur_pos_signal = abs(fft(pos_signal))
        cur_pos_signal = cur_pos_signal[0:int(n_fft)]
        cur_pos_signal = cur_pos_signal / len(cur_pos_signal)
        if count_frames == 1:
            xprev = cur_pos_signal.copy()
        cur_vf = numpy.zeros((total_num_of_features, 1))
        cur_vf[0] = st_zcr(pos_signal)
        cur_vf[1] = st_energy(pos_signal)
        cur_vf[2] = st_energy_entropy(pos_signal)
        [cur_vf[3], cur_vf[4]] = st_spectral_centroid_and_spread(cur_pos_signal, fs)
        cur_vf[5] = st_spectral_entropy(cur_pos_signal)
        cur_vf[6] = st_spectral_flux(cur_pos_signal, xprev)
        cur_vf[7] = st_pectral_roll_off(cur_pos_signal, 0.90, fs)
        cur_vf[num_of_time_spectral_features:num_of_time_spectral_features+nceps, 0] = \
            st_mfcc(cur_pos_signal, fbank, nceps).copy()
        _, chroma_f = st_chroma_features(cur_pos_signal, n_chroma, n_freqs_per_chroma)
        cur_vf[num_of_time_spectral_features + nceps:
               num_of_time_spectral_features + nceps + num_of_chroma_features - 1] = chroma_f
        cur_vf[num_of_time_spectral_features + nceps + num_of_chroma_features - 1] = chroma_f.std()
        st_features.append(cur_vf)
        xprev = cur_pos_signal.copy()

    st_features = numpy.concatenate(st_features, 1)
    return st_features


def st_zcr(frame):
    """
    过零率
    """
    count = len(frame)
    count_z = numpy.sum(numpy.abs(numpy.diff(numpy.sign(frame)))) / 2
    return numpy.float64(count_z) / numpy.float64(count-1.0)


def st_energy(frame):
    """
    短时能量
    """
    return numpy.sum(frame ** 2) / numpy.float64(len(frame))


def st_energy_entropy(frame, num_of_short_blocks=10):
    """
    短时能量熵
    """
    eol = numpy.sum(frame ** 2)
    len_frame = len(frame)
    sub_win_length = int(numpy.floor(len_frame / num_of_short_blocks))
    if len_frame != sub_win_length * num_of_short_blocks:
        frame = frame[0:sub_win_length * num_of_short_blocks]
    sub_windows = frame.reshape(sub_win_length, num_of_short_blocks, order='F').copy()
    sum_windows = numpy.sum(sub_windows ** 2, axis=0) / (eol + EPS)
    neg_entropy = numpy.sum(sum_windows * numpy.log2(sum_windows + EPS))
    return neg_entropy if neg_entropy else -1


def st_spectral_centroid_and_spread(cur_pos_signal, fs):
    """
    短时谱速度
    """
    ind = (numpy.arange(1, len(cur_pos_signal) + 1)) * (fs/(2.0 * len(cur_pos_signal)))
    cur_signal = cur_pos_signal.copy()
    cur_signal = cur_signal / cur_signal.max()
    num = numpy.sum(ind * cur_signal)
    den = numpy.sum(cur_signal) + EPS
    cen = (num / den)
    sum_windows = numpy.sqrt(numpy.sum(((ind - cen) ** 2) * cur_signal) / den)
    cen = cen / (fs / 2.0)
    sum_windows = sum_windows / (fs / 2.0)
    return cen, sum_windows


def st_spectral_entropy(cur_pos_signal, num_of_short_blocks=10):
    """
    短时谱熵
    """
    len_frame = len(cur_pos_signal)
    eol = numpy.sum(cur_pos_signal ** 2)
    sub_win_length = int(numpy.floor(len_frame / num_of_short_blocks))
    if len_frame != sub_win_length * num_of_short_blocks:
        cur_pos_signal = cur_pos_signal[0:sub_win_length * num_of_short_blocks]
    sub_windows = cur_pos_signal.reshape(sub_win_length, num_of_short_blocks,
                                         order='F').copy()
    sum_windows = numpy.sum(sub_windows ** 2, axis=0) / (eol + EPS)
    ne_entropy = numpy.sum(sum_windows*numpy.log2(sum_windows + EPS))
    return -ne_entropy if ne_entropy else -1


def st_spectral_flux(cur_pos_signal, xprev):
    """
    短时谱通量
    """
    sum_x = numpy.sum(cur_pos_signal + EPS)
    sum_prev_x = numpy.sum(xprev + EPS)
    st_flux = numpy.sum((cur_pos_signal / sum_x - xprev/sum_prev_x) ** 2)
    return st_flux


def st_pectral_roll_off(cur_pos_signal, cen, fs):
    """
    短时谱滚动量
    """
    total_energy = numpy.sum(cur_pos_signal ** 2)
    fft_length = len(cur_pos_signal)
    thres = cen*total_energy
    cum_sum = numpy.cumsum(cur_pos_signal ** 2) + EPS
    [no_neg, ] = numpy.nonzero(cum_sum > thres)
    if no_neg[0]:
        m_c = numpy.float64(no_neg[0]) / (float(fft_length))
    else:
        m_c = 0.0
    return m_c


def mfcc_init_filter_banks(fs, nfft):
    """
    mfcc初始滤波宽度
    """
    lowfreq = 133.33
    linsc = 200 / 3.
    logsc = 1.0711703
    num_lin_filt_total = 13
    num_log_filt = 27

    n_filt_total = num_lin_filt_total + num_log_filt
    freqs = numpy.zeros(n_filt_total + 2)
    freqs[:num_lin_filt_total] = lowfreq + numpy.arange(num_lin_filt_total) * linsc
    freqs[num_lin_filt_total:] = freqs[num_lin_filt_total - 1] * \
                                 logsc ** numpy.arange(1, num_log_filt + 3)
    heights = 2. / (freqs[2:] - freqs[0:-2])
    fbank = numpy.zeros((int(n_filt_total), int(nfft)))
    nfreqs = numpy.arange(nfft) / (1. * nfft) * fs

    for i in range(n_filt_total):
        low_tr_freq = freqs[i]
        cen_tr_freq = freqs[i + 1]
        high_tr_freq = freqs[i + 2]

        lid = numpy.arange(numpy.floor(low_tr_freq * nfft / fs) + 1,
                           numpy.floor(cen_tr_freq * nfft / fs) + 1, dtype=numpy.int)
        lslope = heights[i] / (cen_tr_freq - low_tr_freq)
        rid = numpy.arange(numpy.floor(cen_tr_freq * nfft / fs) + 1,
                           numpy.floor(high_tr_freq * nfft / fs) + 1, dtype=numpy.int)
        rslope = heights[i] / (high_tr_freq - cen_tr_freq)
        fbank[i][lid] = lslope * (nfreqs[lid] - low_tr_freq)
        fbank[i][rid] = rslope * (high_tr_freq - nfreqs[rid])

    return fbank, freqs


def st_mfcc(cur_pos_signal, fbank, nceps):
    """
    短时mfcc
    """
    mspec = numpy.log10(numpy.dot(cur_pos_signal, fbank.T) + EPS)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:nceps]
    return ceps


def st_chroma_features_init(nfft, fs):
    """
    色度初始化
    """
    freqs = numpy.array([((st_flux + 1) * fs) / (2 * int(nfft)) for st_flux in range(int(nfft))])
    c_p = 27.50
    n_chroma = numpy.round(12.0 * numpy.log2(freqs / c_p)).astype(int)
    n_freqs_per_chroma = numpy.zeros((n_chroma.shape[0],))
    u_chroma = numpy.unique(n_chroma)
    for u_ch in u_chroma:
        idx = numpy.nonzero(n_chroma == u_ch)
        n_freqs_per_chroma[idx] = idx[0].shape
    return n_chroma, n_freqs_per_chroma


def st_chroma_features(cur_pos_signal, n_chroma, n_freqs_per_chroma):
    """
    短时色度
    """
    chroma_names = ['A', 'A#', 'B', 'cen', 'cen#', 'D', 'D#', 'E', 'st_flux', 'st_flux#', 'G', 'G#']
    spec = cur_pos_signal ** 2
    if n_chroma.max() < n_chroma.shape[0]:
        cen = numpy.zeros((n_chroma.shape[0],))
        cen[n_chroma] = spec
        cen /= n_freqs_per_chroma[n_chroma]
    else:
        no_0_pos = numpy.nonzero(n_chroma > n_chroma.shape[0])[0][0]
        cen = numpy.zeros((n_chroma.shape[0],))
        cen[n_chroma[0:no_0_pos - 1]] = spec
        cen /= n_freqs_per_chroma
    new_d = int(numpy.ceil(cen.shape[0] / 12.0) * 12)
    cur_two = numpy.zeros((new_d,))
    cur_two[0:cen.shape[0]] = cen
    cur_two = cur_two.reshape(int(cur_two.shape[0] / 12), 12)
    final_c = numpy.matrix(numpy.sum(cur_two, axis=0)).T
    final_c /= spec.sum()
    return chroma_names, final_c
