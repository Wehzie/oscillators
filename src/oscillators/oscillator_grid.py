import oscillators.optimization.meta_target as meta_target
import oscillators.optimization.data_analysis as data_analysis

from typing import List

import matplotlib.pyplot as plt
from matplotlib import figure
import numpy as np

class OscillatorGrid:
    """a grid of oscillators"""

    def __init__(self, oscillators: List[meta_target.MetaTarget]) -> None:
        self.oscillators = oscillators

    def animate(self, frame: int, num_frames: int, fig: figure.Figure, lines: List[plt.Line2D]) -> List:
        """return the signal at a given frame index"""
        animation_phase_shift = (frame / num_frames) * 2 * np.pi
        ax_index = 0
        for osc, line in zip(self.oscillators, lines):
            shift_amount = int(animation_phase_shift / (2 * np.pi) * len(osc.signal))
            shifted_signal = np.roll(osc.signal, -shift_amount)
            line.set_data(osc.time, shifted_signal)
            ax = data_analysis.select_axis(fig.axes, len(self.oscillators), ax_index)
            ax.set_xlim(osc.time.min(), osc.time.max())  # set x limits
            y_pad = 0.05
            ax.set_ylim(shifted_signal.min() - y_pad, shifted_signal.max() + y_pad)
            ax_index += 1
        # TODO: x and y limits
        fig.canvas.draw()
        return lines