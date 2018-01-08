#!/usr/bin/env python
from base import mfcc
from base import delta
from base import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("sounds/english.wav")
mfcc_feat = mfcc(sig, rate)
d_mfcc_feat = delta(mfcc_feat, 2)
fbank_feat = logfbank(sig, rate)

print(fbank_feat[1:3,:])
