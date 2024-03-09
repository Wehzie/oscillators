"""
This module implements the abstract MetaTarget class and subclasses.

The MetaTarget class bundles a signal with its time axis and other metadata.
The subclasses implement different ways to load a signal from file or generate a signal.
"""

import oscillators.const as const

from typing import Final, List, Tuple, Union
from pathlib import Path
from abc import ABC

import numpy as np
from scipy import signal
from matplotlib import figure
import matplotlib.pyplot as plt


#### #### #### #### REAL WORLD TARGETS #### #### #### ####


class AbstractTarget(ABC):
    """abstract base class for all meta targets"""

    def __init__(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        sampling_rate: int,
        dtype: np.dtype,
        duration: float,
        name: str,
    ) -> None:
        self.signal = signal
        self.time = time
        self.sampling_rate = sampling_rate
        self.dtype = dtype
        self.duration = duration
        self.samples = len(signal)
        self.name = name

    def __repr__(self) -> str:
        return f"MetaTarget({self.name}, sampling_rate={self.sampling_rate}, dtype={self.dtype}, duration={self.duration})"

    def get_max_freq(self) -> float:
        """return the maximum frequency in the signal"""
        if hasattr(self, "max_freq"):
            return self.max_freq
        else:
            raise NotImplementedError("max_freq is not implemented for this target")
            # return data_analysis.get_max_freq_from_fft(self.signal, 1 / self.sampling_rate)

    def animate(self, frame: int, num_frames: int, fig: figure.Figure, line: plt.Line2D) -> List:
        """return the signal at a given frame index"""
        animation_phase_shift = (frame / num_frames) * 2 * np.pi
        shift_amount = int(animation_phase_shift / (2 * np.pi) * len(self.signal))
        shifted_signal = np.roll(self.signal, -shift_amount)
        y_pad = 0.05
        fig.gca().set_xlim(self.time.min(), self.time.max())  # set x limits
        fig.gca().set_ylim(shifted_signal.min() - y_pad, shifted_signal.max() + y_pad)  # set y limits
        line.set_data(self.time, shifted_signal)
        fig.canvas.draw()
        return line,


#### #### #### #### SYNTHETIC TARGETS #### #### #### ####


class SyntheticTarget(AbstractTarget):
    """abstract class for synthetic target signals"""

    def __init__(
        self,
        duration: float,
        sampling_rate: Union[int, None] = None,
        samples: Union[int, None] = None,
        max_freq: Union[float, None] = None,
    ) -> None:
        """abstract class for synthetic targets

        args:
            duration: the duration of the signal in seconds
            sampling_rate: the sampling rate of the signal
            samples: the number of samples in the signal
            max_freq: the maximum frequency in the signal
        """
        self.dtype = np.float64
        self.duration = duration
        self.derive_samples_or_sampling_rate(duration, samples, sampling_rate, max_freq)
        self.time = np.linspace(0, duration, self.samples, endpoint=False)
        self.max_freq = max_freq

    def compute_oversampling_rate(self, max_freq: float) -> int:
        """compute the oversampling rate for the given sampling rate

        args:
            max_freq: the maximum frequency in the signal
        """
        nyquist_rate = max_freq * 2
        return np.around(nyquist_rate * const.OVERSAMPLING_FACTOR).astype(int)

    def compute_nyquist_rate(self, max_freq: float) -> int:
        """compute the nyquist rate for a given signal

        args:
            max_freq: the maximum frequency in the signal
        """
        nyquist_rate = max_freq * 2
        return nyquist_rate

    def derive_samples_or_sampling_rate(
        self, duration: float, samples: int, sampling_rate: int, max_freq: float
    ) -> None:
        """given a duration infer the number of samples samples or sampling rate"""
        if duration and sampling_rate and max_freq:
            assert sampling_rate >= self.compute_nyquist_rate(
                max_freq
            ), "sampling rate is too low for the given max frequency"
            self.sampling_rate = sampling_rate
            self.samples = np.around(self.sampling_rate * duration).astype(int)
            return

        if duration and sampling_rate:
            self.sampling_rate = sampling_rate
            self.samples = np.around(self.sampling_rate * duration).astype(int)
            return

        if samples and max_freq:
            self.samples = samples
            self.sampling_rate = np.around(samples * 1 / duration).astype(int)
            return

        assert (
            len(set([samples, sampling_rate, max_freq])) == 2
        ), "only one of samples, sampling rate or max_freq should be specified at this point"
        if max_freq:
            self.sampling_rate = self.compute_nyquist_rate(max_freq)
            self.samples = np.around(self.sampling_rate * duration).astype(int)
        elif samples:
            self.samples = samples
            self.sampling_rate = np.around(self.samples * 1 / duration).astype(int)
        elif sampling_rate:  # sampling rate is not None
            self.sampling_rate = sampling_rate
            self.samples = np.around(self.sampling_rate * duration).astype(int)
        else:
            raise ValueError(
                "checks have failed: only one of samples, sampling rate or max_freq should be specified"
            )
    
    def pad_zero(self, short: np.ndarray, len_long: int) -> np.ndarray:
        """evenly zero-pad a short signal up to the desired length"""
        # evenly pad with zeros
        to_pad = len_long - len(short)

        # deal with an odd number of padding so that dimensions are exact
        to_pad_odd = 0
        if to_pad % 2 == 1:
            to_pad_odd = 1

        padded = np.pad(short, (to_pad // 2, to_pad // 2 + to_pad_odd), mode="constant")
        return padded

    def moving_average(self, arr: np.ndarray, window_length: int) -> np.ndarray:
        """compute the l-point average over the signal using a convolution"""
        unpadded = np.convolve(arr, np.ones(window_length), "valid") / window_length
        return self.pad_zero(unpadded, len(arr))


class SineTarget(SyntheticTarget):
    """a sine wave time-series"""

    def __init__(
        self,
        duration: float,
        sampling_rate: Union[int, None] = None,
        samples: Union[int, None] = None,
        freq: float = 1,
        amplitude: float = 1,
        phase: float = 0,
        offset: float = 0,
        name: str = "sine",
    ) -> None:
        super().__init__(duration, sampling_rate, samples, freq)
        self.name = name
        self.signal = np.sin(2 * np.pi * freq * self.time + phase * np.pi) * amplitude + offset

class TriangleTarget(SyntheticTarget):
    """a triangle wave signal"""

    def __init__(
        self,
        duration: float,
        sampling_rate: Union[int, None] = None,
        samples: Union[int, None] = None,
        freq: float = 1,
        amplitude: float = 1,
        phase: float = 0,
        offset: float = 0,
    ) -> None:
        super().__init__(duration, sampling_rate, samples, freq)
        self.name = "triangle"
        self.signal = (
            signal.sawtooth(2 * np.pi * freq * self.time + phase * np.pi, width=0.5) * amplitude
            + offset
        )


class SquareTarget(SyntheticTarget):
    """a square wave signal"""

    def __init__(
        self,
        duration: float,
        sampling_rate: Union[int, None] = None,
        samples: Union[int, None] = None,
        freq: float = 1,
        amplitude: float = 1,
        phase: float = 0,
        offset: float = 0,
    ) -> None:
        super().__init__(duration, sampling_rate, samples, freq)
        self.name = "square"
        self.signal = (
            np.sign(np.sin(2 * np.pi * freq * self.time + phase * np.pi)) * amplitude + offset
        )


class SawtoothTarget(SyntheticTarget):
    """a sawtooth signal"""

    def __init__(
        self,
        duration: float,
        sampling_rate: Union[int, None] = None,
        samples: Union[int, None] = None,
        freq: float = 1,
        amplitude: float = 1,
        phase: float = 0,
        offset: float = 0,
    ) -> None:
        super().__init__(duration, sampling_rate, samples, freq)
        self.name = "sawtooth"
        self.signal = (
            signal.sawtooth(2 * np.pi * freq * self.time + phase * np.pi) * amplitude + offset
        )


class InverseSawtoothTarget(SyntheticTarget):
    """a time-series of the inverse sawtooth signal"""

    def __init__(
        self,
        duration: float,
        sampling_rate: Union[int, None] = None,
        samples: Union[int, None] = None,
        freq: float = 1,
        amplitude: float = 1,
        phase: float = 0,
        offset: float = 0,
    ) -> None:
        super().__init__(duration, sampling_rate, samples, freq)
        self.name = "inverse sawtooth"
        self.signal = (
            -signal.sawtooth(2 * np.pi * freq * self.time + phase * np.pi) * amplitude + offset
        )


class ChirpTarget(SyntheticTarget):
    """a chirp signal is a signal whose frequency increases or decreases over time"""

    def __init__(
        self,
        duration: float,
        sampling_rate: Union[int, None] = None,
        samples: Union[int, None] = None,
        start_freq: float = 0.1,
        stop_freq: Union[float, None] = None,
        amplitude: float = 1,
        offset: float = 0,
        name: str = "chirp",
    ) -> None:
        """
        initialize a chirp signal.

        args:
            duration: the duration of the signal in seconds
            sampling_rate: the sampling rate of the signal,
                           when None the sampling rate is derived from the maximum frequency in the signal
            samples: the number of samples in the signal,
                     when None the number of samples is derived from the sampling rate and duration
            start_freq: the start frequency of the chirp
            stop_freq: the stop frequency of the chirp,
                       when None the stop frequency is derived from the sampling rate
        """
        super().__init__(duration, sampling_rate, samples, stop_freq)
        self.name = name
        if stop_freq is None:
            stop_freq = self.sampling_rate / 20
        self.signal = (
            signal.chirp(self.time, start_freq, self.duration, stop_freq) * amplitude + offset
        )
        self.start_freq = start_freq
        self.stop_freq = stop_freq


class BeatTarget(SyntheticTarget):
    """a beat target is the sum of two sinusoids with different frequencies"""

    def __init__(
        self,
        duration: float,
        sampling_rate: Union[int, None] = None,
        samples: Union[int, None] = None,
        base_freq: float = 1,
        base_freq_factor: float = 2,
        amplitude: float = 1,
        phase: float = 0,
        offset: float = 0,
    ) -> None:
        """
        generate a beat note signal that is the sum of two sinusoids with different frequencies

        args:
            base_freq: the first of two frequency components of the beat note
            base_freq_factor: the frequency of the second frequency components is the product of the base_frequency and the base_freq_factor
        """
        super().__init__(duration, sampling_rate, samples, base_freq)
        self.name = "beat"
        derived_freq = base_freq * base_freq_factor
        self.signal = (
            np.sin(2 * np.pi * base_freq * self.time + phase * np.pi)
            + np.sin(2 * np.pi * derived_freq * self.time + phase * np.pi)
        ) * amplitude + offset


class DampChirpTarget(SyntheticTarget):
    """a damped chirp signal"""

    def __init__(
        self,
        duration: float,
        sampling_rate: Union[int, None] = None,
        samples: Union[int, None] = None,
        start_freq: float = 0.1,
        stop_freq: Union[float, None] = None,
        amplitude: float = 1,
        offset: float = 0,
        name: str = "d. chirp",
    ) -> None:
        """
        initialize a chirp signal.

        args:
            duration: the duration of the signal in seconds
            sampling_rate: the sampling rate of the signal,
                           when None the sampling rate is derived from the maximum frequency in the signal
            samples: the number of samples in the signal,
                     when None the number of samples is derived from the sampling rate and duration
            start_freq: the start frequency of the chirp
            stop_freq: the stop frequency of the chirp,
                       when None the stop frequency is derived from the sampling rate
        """
        super().__init__(duration, sampling_rate, samples, stop_freq)
        self.name = name
        if stop_freq is None:
            stop_freq = self.sampling_rate / 20
        self.signal = (
            signal.chirp(self.time, start_freq, self.duration, stop_freq) * amplitude + offset
        )
        self.signal = self.signal * np.exp(-self.time) ** (1 / self.duration)
        self.start_freq = start_freq
        self.stop_freq = stop_freq


class SmoothGaussianNoiseTarget(SyntheticTarget):
    """a time-series of noise drawn from a gaussian distribution to which a moving average is applied"""

    def __init__(
        self,
        duration: float,
        sampling_rate: Union[int, None] = None,
        samples: Union[int, None] = None,
        amplitude: float = 1,
        offset: float = 0,
        avg_window: int = 10,
    ) -> None:
        super().__init__(duration, sampling_rate, samples)
        self.name = "smooth gaussian noise"
        self.signal = const.RNG.normal(0, 1, self.samples) * amplitude + offset
        self.signal = self.moving_average(self.signal, avg_window)


class SmoothUniformNoiseTarget(SyntheticTarget):
    """a time-series of noise drawn from a uniform distribution to which a moving average is applied"""

    def __init__(
        self,
        duration: float,
        sampling_rate: Union[int, None] = None,
        samples: Union[int, None] = None,
        amplitude: float = 1,
        offset: float = 0,
        avg_window: int = 10,
    ) -> None:
        super().__init__(duration, sampling_rate, samples)
        self.name = "smooth uniform noise"
        self.signal = const.RNG.uniform(-1, 1, self.samples) * amplitude + offset
        self.signal = self.moving_average(self.signal, avg_window)


class GaussianNoiseTarget(SyntheticTarget):
    """a time-series of noise drawn from a Gaussian distribution"""

    def __init__(
        self,
        duration: float,
        sampling_rate: Union[int, None] = None,
        samples: Union[int, None] = None,
        amplitude: float = 1,
        offset: float = 0,
    ) -> None:
        super().__init__(duration, sampling_rate, samples)
        self.name = "gaussian noise"
        self.signal = const.RNG.normal(0, 1, self.samples) * amplitude + offset


class UniformNoiseTarget(SyntheticTarget):
    """a time-series of noise drawn from a uniform distribution"""

    def __init__(
        self,
        duration: float,
        sampling_rate: Union[int, None] = None,
        samples: Union[int, None] = None,
        amplitude: float = 1,
        offset: float = 0,
    ) -> None:
        super().__init__(duration, sampling_rate, samples)
        self.name = "uniform noise"
        self.signal = const.RNG.uniform(-1, 1, self.samples) * amplitude + offset


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t = SineTarget(duration=2e-5, sampling_rate=2e7, freq=300000)
    _ = plt.figure()
    plt.plot(t.time, t.signal, label=t.name, linewidth=3)
    plt.xlabel("time [s]")
    plt.ylabel("amplitude [a.u.]")
    plt.savefig("damp_chirp.png", dpi=300)
    plt.show()
