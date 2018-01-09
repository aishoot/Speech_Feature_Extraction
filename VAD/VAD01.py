import wave
import numpy as np
import matplotlib.pyplot as plt
import Volume as vp

def findIndex(vol,thres):
    l = len(vol)
    ii = 0
    index = np.zeros(4,dtype=np.int16)
    for i in range(l-1):
        if((vol[i]-thres)*(vol[i+1]-thres)<0):
            index[ii]=i
            ii = ii+1
    return index[[0,-1]]

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
threshold1 = max(vol)*0.10
threshold2 = min(vol)*10.0
threshold3 = max(vol)*0.05+min(vol)*5.0

time = np.arange(0,nframes) * (1.0/framerate)
vols = np.arange(0,len(vol)) * (nframes*1.0/len(vol)/framerate)
index1 = findIndex(vol,threshold1)*(nframes*1.0/len(vol)/framerate)
index2 = findIndex(vol,threshold2)*(nframes*1.0/len(vol)/framerate)
index3 = findIndex(vol,threshold3)*(nframes*1.0/len(vol)/framerate)
end = nframes * (1.0/framerate)

plt.subplot(211)
plt.title("VAD01 using volume")
plt.plot(time,waveData,color="black")
plt.plot([index1,index1],[-1,1],'-r')
plt.plot([index2,index2],[-1,1],'-g')
plt.plot([index3,index3],[-1,1],'-b')
plt.ylabel('Amplitude')

plt.subplot(212)
plt.plot(vols,vol,color="black")
plt.plot([0,end],[threshold1,threshold1],'-r', label="threshold 1")
plt.plot([0,end],[threshold2,threshold2],'-g', label="threshold 2")
plt.plot([0,end],[threshold3,threshold3],'-b', label="threshold 3")
plt.legend()
plt.ylabel('Volume(absSum)')
plt.xlabel('time(seconds)')
plt.savefig("VAD01")
plt.show()
