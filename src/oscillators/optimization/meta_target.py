"""
This module implements the abstract MetaTarget class and subclasses.

The MetaTarget class bundles a signal with its time axis and other metadata.
The subclasses implement different ways to load a signal from file or generate a signal.
"""

from . import data_io
from . import data_preprocessor
from . import const
from . import data_analysis

from typing import Final, Type, Union, List
from pathlib import Path
from abc import ABC

import numpy as np
from scipy import signal
from scipy import interpolate
from matplotlib import figure
import matplotlib.pyplot as plt

#### #### #### #### REAL WORLD TARGETS #### #### #### ####


class MetaTarget(ABC):
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
            return data_analysis.get_max_freq_from_fft(self.signal, 1 / self.sampling_rate)

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
        return line

class MetaTargetSample(MetaTarget):
    """load signal from file, save meta data; use sample based processing"""

    def __init__(self, samples: int, name: str, path: Path) -> None:
        self.name = name
        # loading and manipulating the target signal
        raw_sampling_rate, raw_target, raw_dtype = data_io.load_data(path)
        # length checks
        if samples > len(raw_target):
            raise ValueError(
                "The desired number of samples is less than contained in the original data"
            )
        # shorten the target
        target_middle = raw_target
        if len(raw_target) // 3 > samples:
            target_middle = data_preprocessor.take_middle_third(raw_target)
        # downsample
        target_resampled: Final = data_preprocessor.downsample_typesafe(
            target_middle, samples
        )
        # save to wav
        data_io.save_signal_to_wav(
            target_middle, raw_sampling_rate, raw_dtype, Path("data/target_downsampled.wav")
        )

        # set state
        self.signal = target_resampled
        self.samples = len(self.signal)
        self.sampling_rate = data_preprocessor.get_sampling_rate_after_resample(
            target_middle, target_resampled, raw_sampling_rate
        )
        self.dtype = raw_dtype
        self.duration = len(target_resampled) / self.sampling_rate
        self.time = np.linspace(
            0,
            self.duration,
            np.around(self.duration * self.sampling_rate).astype(int),
            endpoint=False,
        )

    def __repr__(self) -> str:
        return f"MetaTargetSample(samples={len(self.signal)}, duration={self.duration}, sampling_rate={self.sampling_rate}, dtype={self.dtype})"


class MetaTargetTime(MetaTarget):
    """load signal from file, save meta data; use time based processing"""

    def __init__(self, time_stop: float, time_start: float, time_step: float, down_sample_factor: float,
                 name: str, path: Path) -> None:
        self.name = name
        # loading and manipulating the target signal
        raw_sampling_rate, raw_target, raw_dtype = data_io.load_data(path)

        # shorten the target to the desired duration
        new_duration = time_stop - time_start  # seconds
        # TODO: comment in when synthetic signals have been refactored
        # assert 1/raw_sampling_rate < new_duration, "The desired duration is less than the sampling rate"
        time = np.arange(0, new_duration, 1 / raw_sampling_rate)
        signal = data_preprocessor.change_duration(raw_target, raw_sampling_rate, new_duration)
        # time returns a non-deterministic number of samples, therefore we need to align the signals
        time, signal = data_preprocessor.align_signals_cut(time, signal)

        self.time = time
        self.signal = signal
        self.sampling_rate = raw_sampling_rate
        self.duration = new_duration
        self.dtype = raw_dtype
        self.samples = len(signal)

        if len(signal) < 1:
            raise ValueError(
                "The selected target duration is less than the distance between two samples"
            )

        # resample the time-aligned signal to the number of samples generated by spice
        spice_samples = data_preprocessor.estimate_spice_samples(time_stop, time_start, time_step, down_sample_factor)
        self.adjust_samples(spice_samples)

    def adjust_samples(self, new_samples: int) -> None:
        """update the signal to the new number of samples"""
        print(f"adjusting samples from {len(self.signal)} to {new_samples}")
        sampling_rate_increase = new_samples / len(self.signal)
        new_sampling_rate = np.around(self.sampling_rate * sampling_rate_increase).astype(int)

        # downsample_typesafe gives a much cleaner signal than scipy.resample
        if new_samples < len(self.signal):
            new_signal = data_preprocessor.downsample_typesafe(
                self.signal, new_samples
            )  # downsample
        else:
            # scipy.resample works well for upsampling
            new_signal = data_preprocessor.resample(self.signal, new_samples)  # resample
            new_signal = new_signal.astype(self.dtype)  # maintain type

        new_time = np.arange(0, self.duration, 1 / new_sampling_rate)
        new_time, new_signal = data_preprocessor.align_signals_cut(new_time, new_signal)

        self.time = new_time
        self.signal = new_signal
        self.sampling_rate = new_sampling_rate
        self.samples = len(new_signal)

    # TODO: should look smoother than the result of scipy.resample with adjust_samples
    def upsample_by_interpolation(
        signal: np.ndarray, sampling_rate: int, new_sampling_rate: int
    ) -> np.ndarray:
        """upsample a signal by interpolation"""
        time = np.arange(0, len(signal) / sampling_rate, 1 / sampling_rate)
        new_time = np.arange(0, len(signal) / sampling_rate, 1 / new_sampling_rate)
        new_signal = interpolate.interp1d(time, signal, kind="cubic")(new_time)
        return new_signal, new_time

    def __repr__(self) -> str:
        return f"MetaTarget(signal={self.signal}, sampling_rate={self.sampling_rate}, dtype={self.dtype})"


class MetaTargetSmart(MetaTarget):
    """infer time or sample based processing based on the arguments"""

    def __init__(
        self,
        signal: np.ndarray,
        time: Union[np.ndarray, None] = None,
        sampling_rate: Union[int, None] = None,
        duration: Union[float, None] = None,
        name: Union[str, None] = None,
    ):
        self.signal = signal
        self.dtype = signal.dtype
        self.samples = len(signal)
        self.name = name if name else "Unnamed Signal"
        if sampling_rate is not None:
            self.sampling_rate = sampling_rate
            self.duration = 1 / sampling_rate * self.samples
            self.time = np.linspace(0, self.duration, self.samples, endpoint=False)
        elif duration is not None:
            self.duration = duration
            self.sampling_rate = np.around(self.samples / duration).astype(int)
            self.time = np.linspace(0, self.duration, self.samples, endpoint=False)
        elif time is not None:
            self.time = time
            self.sampling_rate = np.around(1 / (time[1] - time[0])).astype(int)
            self.duration = time[-1] - time[0]
        else:
            raise ValueError("either time, sampling rate or duration must be provided")


#### #### #### #### SYNTHETIC TARGETS #### #### #### ####


class SyntheticTarget(MetaTarget):
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

    def moving_average(self, arr: np.ndarray, window_length: int) -> np.ndarray:
        """compute the l-point average over the signal using a convolution"""
        unpadded = np.convolve(arr, np.ones(window_length), "valid") / window_length
        return data_preprocessor.pad_zero(unpadded, len(arr))
    
    @staticmethod
    def map_name_to_class(name: str) -> 'Type[SyntheticTarget]':
        """map a name to a synthetic target class"""
        return {
            "sine": SineTarget,
            "triangle": TriangleTarget,
            "square": SquareTarget,
            "sawtooth": SawtoothTarget,
            "inverse sawtooth": InverseSawtoothTarget,
            "chirp": ChirpTarget,
            "beat": BeatTarget,
            "damp chirp": DampChirpTarget,
            "smooth gaussian noise": SmoothGaussianNoiseTarget,
            "smooth uniform noise": SmoothUniformNoiseTarget,
            "gaussian noise": GaussianNoiseTarget,
            "uniform noise": UniformNoiseTarget,
        }[name]


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
        base_freq_factor: float = 1 / 10,
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
