import wave
import numpy as np
import matplotlib.pyplot as plt
import Volume as vp
from ZeroCR import ZeroCR

fw = wave.open('../sounds/sunday.wav','r')
params = fw.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
strData = fw.readframes(nframes)
waveData = np.fromstring(strData, dtype=np.int16)
waveData = waveData*1.0/max(abs(waveData))  # normalization
fw.close()

frameSize = 256
overLap = 128
vol = vp.calVolume(waveData,frameSize,overLap)
zcr = ZeroCR(waveData,frameSize,overLap)
threshold1 = max(vol)*0.10
threshold2 = min(vol)*10.0
threshold3 = max(vol)*0.05+min(vol)*5.0
threshold12 = max(zcr)*0.10
threshold22 = min(zcr)*10.0
threshold32 = max(zcr)*0.05+min(zcr)*5.0

time = np.arange(0,nframes) * (1.0/framerate)
vols = np.arange(0,len(vol)) * (nframes*1.0/len(vol)/framerate)
zcrs = np.arange(0,len(zcr))*(nframes*1.0/len(zcr)/framerate)
end = nframes * (1.0/framerate)
plt.subplot(311)
plt.title("VAD02 using volume and ZeroCR")
plt.plot(time,waveData,color="black")
plt.ylabel('Amplitude')

plt.subplot(312)
plt.plot(vols,vol,color="black")
plt.plot([0,end],[threshold1,threshold1],'-r', label="threshold 1")
plt.plot([0,end],[threshold2,threshold2],'-g', label="threshold 2")
plt.plot([0,end],[threshold3,threshold3],'-b', label="threshold 3")
plt.legend()
plt.ylabel('Volume(absSum)')

plt.subplot(313)
plt.plot(zcrs,zcr,color="black")
plt.plot([0,end],[threshold12,threshold12],'-r', label="threshold 12")
plt.plot([0,end],[threshold22,threshold22],'-g', label="threshold 22")
plt.plot([0,end],[threshold32,threshold32],'-b', label="threshold 32")
plt.legend()
plt.ylabel('Zero-Crossing Rate')
plt.xlabel('time(seconds)')
plt.savefig("VAD02")
plt.show()
