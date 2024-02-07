import unittest
import numpy as np
from audio_engine import mfcc


class TestMFCC(unittest.TestCase):

    def test_basic_output_shape(self):
        """ Test that the MFCC output has the correct shape. """
        signal = np.random.rand(16000)  # 1 second of audio sampled at 16 kHz
        samplerate = 16000  # Hz
        winlen = 0.025  # 25 ms
        winstep = 0.01  # 10 ms
        numcep = 13  # typically used number of cepstra

        expected_num_frames = int(np.ceil((len(signal) - samplerate * winlen) / (samplerate * winstep))) + 1
        mfcc_features = mfcc(signal, samplerate, winlen, winstep, numcep=numcep)

        self.assertEqual(mfcc_features.shape, (expected_num_frames, numcep))

    def test_zero_signal(self):
        """ Test that the MFCC function returns an empty array when given a zero-length signal. """
        signal = np.array([])  # empty signal
        samplerate = 16000  # Hz

        mfcc_features = mfcc(signal, samplerate)

        self.assertEqual(mfcc_features.size, 0)

    def test_short_signal(self):
        """ Test that the MFCC function can handle a signal shorter than a single frame. """
        signal = np.random.rand(200)  # very short signal
        samplerate = 16000  # Hz

        mfcc_features = mfcc(signal, samplerate)

        self.assertGreater(mfcc_features.size, 0)  # Should still produce output

    def test_parameter_variations(self):
        """ Test the MFCC function with various parameter combinations. """
        signal = np.random.rand(16000)  # 1 second of audio
        samplerate = 16000  # Hz

        # Test a variety of parameters
        for winlen in [0.015, 0.020, 0.025]:
            for winstep in [0.005, 0.01, 0.015]:
                for numcep in [10, 13, 16]:
                    with self.subTest(winlen=winlen, winstep=winstep, numcep=numcep):
                        mfcc_features = mfcc(signal, samplerate, winlen=winlen, winstep=winstep, numcep=numcep)
                        expected_num_frames = int(np.ceil((len(signal) - samplerate * winlen) / (samplerate * winstep))) + 1
                        self.assertEqual(mfcc_features.shape, (expected_num_frames, numcep))

    # Add more test cases as needed for different signal types, parameter edge cases, etc.

if __name__ == '__main__':
    unittest.main()
