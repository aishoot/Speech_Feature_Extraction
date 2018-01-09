import wave
import numpy as np
import pylab as pl
import ZeroCR

# read wave file and get parameters.
fw = wave.open('../sounds/aeiou.wav','rb')
params = fw.getparams()
print(params)
nchannels, sampwidth, framerate, nframes = params[:4]
strData = fw.readframes(nframes)
waveData = np.fromstring(strData, dtype=np.int16)
waveData = waveData*1.0/max(abs(waveData))  # normalization
fw.close()

# calculate Zero Cross Rate
frameSize = 256
overLap = 0
zcr = ZeroCR.ZeroCR(waveData,frameSize,overLap)

# plot the wave
time = np.arange(0, len(waveData)) * (1.0 / framerate)
time2 = np.arange(0, len(zcr))*(len(waveData)*1.0/len(zcr)/framerate)
pl.subplot(211)
pl.plot(time, waveData)
pl.ylabel("Amplitude")

pl.subplot(212)
pl.plot(time2, zcr)
pl.ylabel("ZCR")
pl.xlabel("time (seconds)")
pl.savefig("ZeroCR.png")
pl.show()
