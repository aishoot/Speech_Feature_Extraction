import math
import numpy as np

# Orignal auto-correlation function(ACF)
def ACF(frame):
    flen = len(frame)
    acf = np.zeros(flen)
    for i in range(flen):
        acf[i] = np.sum(frame[i:flen]*frame[0:flen-i])
    return acf

# ACF with weight
def ACF2(frame):
    flen = len(frame)
    acf = np.zeros(flen)
    for i in range(flen):
        acf[i] = np.sum(frame[i:flen]*frame[0:flen-i])/(flen-i)
    return acf

# ACF to half frame length
def ACF3(frame):
    flen = len(frame)
    acf = np.zeros(flen/2)
    for i in range(flen/2):
        acf[i] = np.sum(frame[i:flen]*frame[0:flen-i])
    return acf

# normalized squared difference function(NSDF)
def NSDF(frame):
    flen = len(frame)
    nsdf = np.zeros(flen)
    for i in range(flen):
        s1 = np.sum(frame[i:flen]*frame[0:flen-i])
        s2 = np.sum(frame[i:flen]*frame[i:flen])
        s3 = np.sum(frame[0:flen-i]*frame[0:flen-i])
        nsdf[i] = 2.0*s1/(s2+s3)
    return nsdf

# AMDF (average magnitude difference function) 
def AMDF(frame):
    flen = len(frame)
    amdf = np.zeros(flen)
    for i in range(flen):
        amdf[i] = -np.sum(np.abs(frame[i:flen]-frame[0:flen-i]))  # to adjust to ACF, I use the -AMDF
    return amdf

# AMDF with weight
def AMDF2(frame):
    flen = len(frame)
    amdf = np.zeros(flen)
    for i in range(flen):
        amdf[i] = -np.sum(np.abs(frame[i:flen]-frame[0:flen-i]))/(flen-i)  # to adjust to ACF, I use the -AMDF
    return amdf

# AMDF to half frame length
def AMDF3(frame):
    flen = len(frame)
    amdf = np.zeros(flen/2)
    for i in range(flen/2):
        amdf[i] = -np.sum(np.abs(frame[i:flen]-frame[0:flen-i]))  # to adjust to ACF, I use the -AMDF
    return amdf

# Pitch Tracking
def PitchTrack(waveData,frameRate,frameSize,overLap,acfmethod):
    wlen = len(waveData)
    step = frameSize - overLap
    frameNum = int(math.ceil(wlen*1.0/step))
    pitchtrack = np.zeros(frameNum)
    for i in range(frameNum):
        #acf = acfmethod(waveData[i*step : i*step+frameSize])  # Bug
        acf = acfmethod(waveData[int(i*step):int(i*step+frameSize)])
        acf[0:30] = np.min(acf)
        acfmax = np.argmax(acf)
        pitchtrack[i] = frameRate*1.0/acfmax
    return pitchtrack
