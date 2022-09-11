import numpy as np
import scipy.signal as sp_sgn


def statistical_wavelet(d, ntwest, dt, nfft=2**10, nsmooth=None):
    """Statistical wavelet estimation.

    Zero-phase wavelet is estimated as average amplitude spectrum of the data
    ``d`` and converted back to time domain.

    Parameters
    ----------
    d : :obj:`np.ndarray.data`
        Data of size `:math:`[ny \times nx \ time nt/nz]` or
        `:math:`[nx \ time nt/nz]`  or `:math:`[nt/nz]`
    ntwest : :obj:`int`
        Number of samples of estimated wavelet
    dt : :obj:`float`
        Sampling step in time/depth axis
    nfft : :obj:`int`, optional
        Number of fft samples
    nsmooth : :obj:`int`, optional
        Number of samples for smoothing of amplitude spectrum
        (if ``None`` smoothing is not applied)

    Returns
    -------
    wav_est : :obj:`np.ndarray.data`
        Estimated wavelet
    wav_est_fft : :obj:`np.ndarray.data`
        Estimated wavelet in frequency domain
    twest : :obj:`np.ndarray.data`
        Time axis
    fwest : :obj:`np.ndarray.data`
        Frequency axis
    wcenter : :obj:`int`
        Index of wavelet center
    """
    ndims = len(d.shape)

    # create time axis
    twest = np.arange(ntwest) * dt
    twest = np.concatenate((np.flipud(-twest[1:]), twest), axis=0)

    # estimate wavelet spectrum
    if ndims == 1:
        wav_est_fft = np.abs(np.fft.fft(d, nfft, axis=-1))
    elif ndims == 2:
        wav_est_fft = np.mean(np.abs(np.fft.fft(d, nfft, axis=-1)), axis=0)
    else:
        wav_est_fft = np.mean(np.abs(np.fft.fft(d, nfft, axis=-1)), axis=(0, 1))
    fwest = np.fft.fftfreq(nfft, d=dt)

    # create wavelet in time
    wav_est = np.real(np.fft.ifft(wav_est_fft)[:ntwest])
    wav_est = np.concatenate((np.flipud(wav_est[1:]), wav_est), axis=0)
    wav_est = wav_est / wav_est.max()
    wcenter = np.argmax(np.abs(wav_est))

    # apply smoothing
    if nsmooth is not None:
        smooth=np.ones(nsmooth)/nsmooth
        wav_est_max= wav_est.max()
        wav_est=sp_sgn.filtfilt(smooth,1,wav_est)
        wav_est= wav_est * (wav_est_max/wav_est.max())

    return wav_est, wav_est_fft[:nfft//2], twest, fwest[:nfft//2], wcenter


