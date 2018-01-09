import wave
import numpy as np
import matplotlib.pyplot as plt

fw = wave.open('../sounds/aeiou.wav','r')
print(fw.getparams())
soundInfo = fw.readframes(-1)
soundInfo = np.fromstring(soundInfo, np.int16)
f = fw.getframerate()
fw.close()

plt.subplot(211)
plt.plot(soundInfo)
plt.ylabel('Amplitude')
plt.title('Wave and spectrogram of aeiou.wav')

plt.subplot(212)
plt.specgram(soundInfo, Fs=f)
plt.ylabel('Frequency')
plt.xlabel('time(seconds)')
plt.savefig("spectrogram.png")
plt.show()