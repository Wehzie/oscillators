from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt


def infer_subplot_rows_cols(n_signals: int) -> Tuple[int, int]:
    """infer the number of rows and columns for a grid of subplots"""
    n_rows = None

    i = 0
    while i**2 <= n_signals:
        n_rows = i
        i += 1
    remainder = n_signals - (i - 1) ** 2
    n_cols = n_rows
    n_rows = n_rows + int(np.ceil(remainder / n_cols))
    return n_rows, n_cols


def plot_multiple_targets_common_axis(
    targets: List, show: bool = False, save_path: Path = None
) -> None:
    """plot multiple targets by aligning to the number of samples in the target with the fewest samples"""
    min_samples = min([target.samples for target in targets])
    time = np.linspace(0, 1, min_samples, endpoint=False)
    target_matrix = np.array([target.signal[:min_samples] for target in targets])
    plot_individual_oscillators(target_matrix, time, show=show, save_path=save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()


def plot_multiple_targets(
    targets: List, show: bool = False, save_path: Path = None
) -> None:
    """plot multiple targets in a grid of subplots; plots are aligned on one time axis"""
    nrows, ncols = infer_subplot_rows_cols(len(targets))
    _, axes = plt.subplots(nrows, ncols, layout="constrained")
    for i, ax in enumerate(axes.flat):
        ax.plot(targets[i].time, targets[i].signal)
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()


def plot_individual_oscillators(
    signal_matrix: np.ndarray,
    time: np.ndarray = None,
    oscillators_per_subplot: Union[None, int] = 25,
    show: bool = False,
    save_path: Path = None,
) -> None:
    """show n individual oscillator signals in a grid of subplots

    args:
        signal_matrix: a matrix of single-oscillator signals
        time: times at which the signals were sampled
        oscillators_per_subplot: number of oscillators per subplot. If None, plot all oscillators in one subplot
        show: show the plot
        save_path: save the plot to a file
    """

    def subset_matrix(signal_matrix: np.ndarray, oscillators_per_subplot: int) -> List[np.ndarray]:
        """split a matrix into subsets of n rows, returns a view"""
        subsets = []
        for row in range(0, signal_matrix.shape[0], oscillators_per_subplot):
            subsets.append(signal_matrix[row : row + oscillators_per_subplot, :])
        return subsets

    def plot_row_per_plot(signal_matrix: np.ndarray, axes: plt.Axes) -> None:
        """plot one signal per row of a matrix in a grid of subplots"""
        n_signals = signal_matrix.shape[0]  # one signal per row
        n_rows, n_cols = infer_subplot_rows_cols(n_signals)

        # plot one signal into each subplot
        signal_counter = 0
        for r in range(n_rows):
            for c in range(n_cols):
                if signal_counter >= n_signals:
                    break
                if time is None:
                    axes[r, c].plot(signal_matrix[signal_counter, :])
                else:
                    axes[r, c].plot(time, signal_matrix[signal_counter, :])
                signal_counter += 1

    def init_axes(signal_matrix: np.ndarray) -> plt.Axes:
        """initialize a grid of subplots"""
        n_signals = signal_matrix.shape[0]
        rows, cols = infer_subplot_rows_cols(n_signals)
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
        return fig, axes

    def add_labels(axes: plt.Axes, signal_matrix: np.ndarray) -> None:
        """add labels to the subplots"""
        n_signals = signal_matrix.shape[0]
        rows, cols = infer_subplot_rows_cols(n_signals)
        plt.suptitle("individual oscillators")
        if time is None:  # samples
            axes[rows - 1, cols // 2].set_xlabel("time [a.u.]")
        else:  # time
            axes[rows - 1, cols // 2].set_xlabel("time [s]")
        axes[rows // 2, 0].set_ylabel("amplitude [a.u.]")

    figures = []
    if oscillators_per_subplot is None:  # plot all oscillators in one figure
        fig, axes = init_axes(signal_matrix)
        figures.append(fig)
        plot_row_per_plot(signal_matrix, axes)
        add_labels(axes, signal_matrix)

    else:  # divide oscillators across figures
        subsets = subset_matrix(signal_matrix, oscillators_per_subplot)
        for i, subset in enumerate(subsets):
            fig, axes = init_axes(subset)
            figures.append(fig)
            plot_row_per_plot(subset, axes)
            add_labels(axes, subset)

    if save_path:  # save all figures generated in this function
        for i, fig in enumerate(figures):
            path = (
                None
                if save_path is None
                else Path(save_path.parent, f"{save_path.stem}_{i}{save_path.suffix}")
            )
            fig.savefig(path, dpi=300)
    if show:
        plt.show()

