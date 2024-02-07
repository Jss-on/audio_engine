#!/usr/bin/env python

from audio_engine import mfcc
from audio_engine import delta
from audio_engine import fbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("audio.wav")
mfcc_feat = mfcc(sig,rate)
d_mfcc_feat = delta(mfcc_feat, 2)
fbank_feat = fbank(sig,rate)

print(fbank_feat)