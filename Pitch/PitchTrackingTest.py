import wave
import numpy as np
import pylab as pl
import PitchTracking as pt

# read wave file and get parameters.
fw = wave.open('../sounds/aeiou.wav','rb')
params = fw.getparams()
print(params)
nchannels, sampwidth, framerate, nframes = params[:4]
strData = fw.readframes(nframes)
waveData = np.fromstring(strData, dtype=np.int16)
waveData = waveData*1.0/max(abs(waveData))  # normalization
fw.close()

# plot the wave
time = np.arange(0, len(waveData)) * (1.0 / framerate)

frameSize = 512
overLap = frameSize/2
idx1 = 10000
idx2 = idx1+frameSize
index1 = idx1*1.0 / framerate
index2 = idx2*1.0 / framerate
acf = pt.ACF(waveData[idx1:idx2])
acf[0:10] = -acf[0]
acfmax = np.argmax(acf)
print(acfmax)
print(framerate*1.0/acfmax)

pl.subplot(411)
pl.title("pitchTrack")
pl.plot(time, waveData)
pl.plot([index1,index1],[-1,1],'r')
pl.plot([index2,index2],[-1,1],'r')
pl.xlabel("time (seconds)")
pl.ylabel("Amplitude")

pl.subplot(412)
pl.plot(np.arange(frameSize),waveData[idx1:idx2],'r')
pl.xlabel("index in 1 frame")
pl.ylabel("Amplitude")

pl.subplot(413)
pl.plot(np.arange(frameSize),acf,'g')
pl.xlabel("index in 1 frame")
pl.ylabel("ACF")

# pitch tracking
acfmethod = pt.ACF
pitchtrack = pt.PitchTrack(waveData, framerate, frameSize, overLap, acfmethod)
xpt = np.arange(0, len(pitchtrack)) *( len(waveData) *1.0/ len(pitchtrack) / framerate )
pl.subplot(414)
pl.plot(xpt,pitchtrack,'-*')
pl.xlabel('time (seconds)')
pl.ylabel('Frequency (Hz)')

#pl.savefig("pitchTrack.png")
pl.show()
