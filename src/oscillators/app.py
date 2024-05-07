import oscillators.optimization.data_analysis as data_analysis
import oscillators.oscillator_grid as oscillator_grid
import oscillators.optimization.const as const
import oscillators.optimization.meta_target as meta_target
import oscillators.optimization.dist as dist
import oscillators.optimization.gen_signal_python as python_generator
import oscillators.optimization.gen_signal_args_types as generator_distribution
import oscillators.optimization.algo_args_type as algo_args
import oscillators.optimization.algo_gradient as algo_gradient
import oscillators.optimization.algo_las_vegas as algo_las_vegas
import oscillators.optimization.algo_monte_carlo as algo_monte_carlo
import oscillators.optimization.sample as sample

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from PIL import Image as PIL_Image
from PIL import ImageDraw, ImageFont
import numpy as np

from pathlib import Path
from typing import List, Union
import threading
import http.server
import socketserver
import os
import tempfile
import threading
from dataclasses import dataclass

PORT = 8003
NUM_FRAMES = 40
SAMPLING_RATE = 100

class oscillators(toga.App):
    def startup(self):
        self.generating_target = False
        self.generating_oscillators = False
        self.generating_prediction = False
        self.on_exit = self.exit_handler

        self.setup_target_webview()
        self.setup_oscillators_webview()
        self.setup_prediction_webview()
        self.setup_controls_bar()
        self.compose_window()

        self.setup_target_plot()
        self.setup_oscillator_plot()
        self.setup_prediction_plot()

        self.init_gifs()
        
        self.start_target_animation()
        self.start_oscillator_animation()
        self.start_prediction_animation()
        self.start_server()

        self.global_counter = 0


    def init_gifs(self):
        self.target_gif = None
        self.oscillator_gif = None
        self.prediction_gif = None # predicted approximation of the target
        self.create_placeholder_gif()

    def create_placeholder_gif(self):
        # Create a placeholder GIF file
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as temp:
            self.placeholder_gif = Path(temp.name)
        img = PIL_Image.new('RGB', (800, 400), color = (255, 255, 255))  # White background
        d = ImageDraw.Draw(img)
        font_size = 60
        # CROSS-PLATFORM: .font_variant not supported on Android
        fnt = ImageFont.load_default()#.font_variant(size=font_size)
        text = "Generating animation."
        # estimate the width and height of the text from number of characters

        text_width, text_height = 1/2 * font_size * len(text), 1/2 * font_size
        x = (img.width - text_width) / 2
        y = (img.height - text_height) / 2
        d.text((x, y), text, font=fnt, fill=(0, 0, 0))
        img.save(self.placeholder_gif, 'GIF')
    
    async def exit_handler(self, app):
        # Return True if app should close, and False if it should remain open
        if await self.main_window.confirm_dialog(
            "Toga", "Are you sure you want to quit?"
        ):
            self.httpd.server_close() # free the port
            self.httpd.shutdown() # stop server forever
            self.server_thread.join() # wait for the server to stop
            if self.target_animation_thread.is_alive():
                self.target_animation_thread.join()
            if self.oscillator_animation_thread.is_alive():
                self.oscillator_animation_thread.join()
            if self.prediction_animation_thread.is_alive():
                self.prediction_animation_thread.join()
            os.remove(self.placeholder_gif) # remove the temporary file

            return True
        else:
            return False
        
    def setup_controls_bar(self):
        self.setup_target_controls()
        self.setup_oscillator_controls()
        spacer = toga.Box(style=Pack(height=30))
        self.controls_bar = toga.Box(children=[self.target_controls, spacer, self.oscillator_controls],
                                     style=Pack(direction=COLUMN))
        
    def setup_oscillator_controls(self):
        pad_left_right = Pack(padding=(0, 10, 0, 10))
        self.n_oscillators_slider = toga.Slider(min=1,
                                                max=12,
                                                value=3,
                                                style=pad_left_right,
                                                on_release=self.update_oscillators_and_prediction)
        self.oscillator_wave_type = toga.Selection(
            items=['Sine', 'Triangle', 'Square', 'Sawtooth', 'Inverse Sawtooth', 'Chirp', 'Beat', 'Damp Chirp',
                   'Smooth Gaussian Noise', 'Smooth Uniform Noise', 'Gaussian Noise', 'Uniform Noise'],
            style=Pack(padding=10),
            on_change=self.update_oscillators_and_prediction
        )
        self.optimization_algorithm = toga.Selection(
            items=["Linear Regression",
                   "Monte Carlo Random Walk", # MCExploit
                   "Las Vegas Random Walk"
                   ],
            style=Pack(padding=10)
        )
        self.perturbations_slider = toga.NumberInput(min=1, max=10000, value=100, style=pad_left_right)
        self.optimize_toggle = toga.Switch(
            text='Optimize',
            value=False,
            style=Pack(padding=(10, 10, 0, 10)),
            on_change=self.optimize_ensemble
        )

        self.oscillator_controls = toga.Box(
            children=[
                toga.Label('Number of oscillators:', style=Pack(padding=(10, 10, 0, 10))),
                self.n_oscillators_slider,
                toga.Label('Wave type:', style=Pack(padding=(10, 10, 0, 10))),
                self.oscillator_wave_type,
                toga.Label('Perturbations:', style=Pack(padding=(10, 10, 0, 10))),
                self.perturbations_slider,
                toga.Label('Optimization algorithm:', style=Pack(padding=(10, 10, 0, 10))),
                self.optimization_algorithm,
                self.optimize_toggle
            ],
            style=Pack(direction=COLUMN)
        )

    def optimize_ensemble(self, widget):
        if self.optimize_toggle.value == True:
            # disable oscillator controls
            self.optimization_algorithm.enabled = False
            self.oscillator_wave_type.enabled = False
            self.n_oscillators_slider.enabled = False
            # disable target controls
            for child in self.target_controls.children:
                child.enabled = False
            # begin optimization
            n_osc = int(self.n_oscillators_slider.value)
            self.generator_distribution = generator_distribution.PythonSignalRandArgs(
                description="test base-parameters for drawing oscillators from a uniform distribution",
                n_osc=n_osc,
                duration=1,
                samples=SAMPLING_RATE,
                freq_dist=dist.Dist(const.RNG.uniform, low=0.1, high=10),
                amplitude=0.5,
                weight_dist=dist.WeightDist(
                    const.RNG.uniform, low=0, high=10, n=n_osc
                ),
                phase_dist=dist.Dist(
                    const.RNG.uniform, low=-1 / 3, high=1 / 3
                ),
                offset_dist=dist.Dist(const.RNG.uniform, low=0, high=0),
                sampling_rate=SAMPLING_RATE,
            )
            self.signal_generator = python_generator.PythonSigGen()
            self.optimizer_args = algo_args.AlgoArgs(
                self.signal_generator,
                self.generator_distribution,
                self.target_plot.target,
                max_z_ops=int(self.perturbations_slider.value))
            OptimizationAlgo = self.select_optimization_algorithm(self.optimization_algorithm.value)
            self.optimizer = OptimizationAlgo(self.optimizer_args)
            # start worker thread
            self.optimize_thread = threading.Thread(target=self.optimize_ensemble_thread)
            self.optimize_thread.start()
            # matplotlib is not thread-safe, therefore a new plot must be created on the main thread
            self.optimize_thread.join()
            self.update_oscillators_and_prediction(self.optimized_ensemble_sum)

        else:
            self.optimization_algorithm.enabled = True
            self.oscillator_wave_type.enabled = True
            self.n_oscillators_slider.enabled = True
            for child in self.target_controls.children:
                child.enabled = True

    def optimize_ensemble_thread(self):
        optimized_ensemble_sum, ops = self.optimizer.search()
        print(f"Optimization finished after {ops} operations")
        self.optimize_toggle.value = False
        self.optimized_ensemble_sum = optimized_ensemble_sum

    def setup_target_controls(self):
        self.wave_type = toga.Selection(
            items=['Sine', 'Triangle', 'Square', 'Sawtooth', 'Inverse Sawtooth', 'Chirp', 'Beat', 'Damp Chirp',
                   'Smooth Gaussian Noise', 'Smooth Uniform Noise', 'Gaussian Noise', 'Uniform Noise'],
            style=Pack(padding=10),
            on_change=self.update_target_parameters
        )
        pad_left_right = Pack(padding=(0, 10, 0, 10))
        self.target_amplitude_slider = toga.Slider(min=0.1, max=10, value=1, on_release=self.update_target_parameters, style=pad_left_right)
        self.target_offset_slider = toga.Slider(min=-5, max=5, value=0, on_release=self.update_target_parameters, style=pad_left_right)
        self.target_phase_slider = toga.Slider(min=0, max=2*np.pi, value=0, on_release=self.update_target_parameters, style=pad_left_right)
        self.target_frequency_slider = toga.Slider(min=0.1, max=10, value=1, on_release=self.update_target_parameters, style=pad_left_right)

        self.target_controls = toga.Box(
            children=[
                self.wave_type,
                toga.Label('Amplitude:', style=Pack(padding=(10, 10, 0, 10))),
                self.target_amplitude_slider,
                toga.Label('Offset:', style=Pack(padding=(10, 10, 0, 10))),
                self.target_offset_slider,
                toga.Label('Phase:', style=Pack(padding=(10, 10, 0, 10))),
                self.target_phase_slider,
                toga.Label('Frequency:', style=Pack(padding=(10, 10, 0, 10))),
                self.target_frequency_slider
            ],
            style=Pack(direction=COLUMN)
        )

    def update_target_parameters(self, *args, **kwargs):
        # Stop the current animation
        self.target_animation.event_source.stop()

        self.target_plot.target = self.select_wave_type(self.wave_type.value,
                                                        self.target_frequency_slider.value,
                                                        self.target_amplitude_slider.value,
                                                        self.target_offset_slider.value,
                                                        self.target_phase_slider.value)

        self.start_target_animation()

    def update_oscillators_and_prediction(self, optimized_ensemble=None, *args, **kwargs):
        self.update_oscillator_parameters(optimized_ensemble)
        self.update_prediction_parameters(optimized_ensemble)

    def update_oscillator_parameters(self, optimized_ensemble=None, *args, **kwargs):
        self.oscillator_plot = None
        self.setup_oscillator_plot(optimized_ensemble)
        self.start_oscillator_animation()

    def update_prediction_parameters(self, optimized_ensemble_sum=None, *args, **kwargs):
        self.prediction_plot = None
        self.setup_prediction_plot(optimized_ensemble_sum)
        self.start_prediction_animation()

    def select_wave_type(self, wave_type: str,
                         freq: float, amplitude: float, offset: float, phase: float,
                         *args, **kwargs) -> meta_target.MetaTarget:
        """Return a target object from a string"""

        # Initialize new target
        if wave_type == 'Sine':
            signal = meta_target.SineTarget(1, SAMPLING_RATE, freq=freq, amplitude=amplitude, offset=offset, phase=phase)
        elif wave_type == 'Triangle':
            signal = meta_target.TriangleTarget(1, SAMPLING_RATE, freq=freq, amplitude=amplitude, offset=offset, phase=phase)
        elif wave_type == "Square":
            signal = meta_target.SquareTarget(1, SAMPLING_RATE, freq=freq, amplitude=amplitude, offset=offset, phase=phase)
        elif wave_type == "Sawtooth":
            signal = meta_target.SawtoothTarget(1, SAMPLING_RATE, freq=freq, amplitude=amplitude, offset=offset, phase=phase)
        elif wave_type == "Inverse Sawtooth":
            signal = meta_target.InverseSawtoothTarget(1, SAMPLING_RATE, freq=freq, amplitude=amplitude, offset=offset, phase=phase)
        elif wave_type == "Chirp":
            signal = meta_target.ChirpTarget(1, SAMPLING_RATE, stop_freq=freq, amplitude=amplitude, offset=offset) # TODO: phase
        elif wave_type == "Beat":
            signal = meta_target.BeatTarget(1, SAMPLING_RATE, base_freq=freq, amplitude=amplitude, offset=offset, phase=phase)
        elif wave_type == "Damp Chirp":
            signal = meta_target.DampChirpTarget(1, SAMPLING_RATE, stop_freq=freq, amplitude=amplitude, offset=offset) # TODO: phase
        elif wave_type == "Smooth Gaussian Noise":
            signal = meta_target.SmoothGaussianNoiseTarget(1, SAMPLING_RATE, amplitude=amplitude, offset=offset) # TODO: window-length
        elif wave_type == "Smooth Uniform Noise":
            signal = meta_target.SmoothUniformNoiseTarget(1, SAMPLING_RATE, amplitude=amplitude, offset=offset) # TODO: window-length
        elif wave_type == "Gaussian Noise":
            signal = meta_target.GaussianNoiseTarget(1, SAMPLING_RATE, amplitude=amplitude, offset=offset)
        elif wave_type == "Uniform Noise":
            signal = meta_target.UniformNoiseTarget(1, SAMPLING_RATE, amplitude=amplitude, offset=offset)        
        else:
            raise ValueError("Invalid wave type")
        return signal
    
    def select_optimization_algorithm(self, algo_type: str, *args, **kwargs):
        if algo_type == "Linear Regression":
            algo = algo_gradient.LinearRegression
        elif algo_type == "Monte Carlo Random Walk":
            algo = algo_monte_carlo.MCExploitWeight
        elif algo_type == "Las Vegas Random Walk":
            algo = algo_las_vegas.LasVegasWeight
        else:
            raise ValueError("Invalid optimization algorithm")
        return algo
    
    def setup_target_plot(self):
        @dataclass
        class TargetPlot:
            fig = Figure()
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            target = meta_target.SineTarget(1, SAMPLING_RATE)
            line = ax.plot([], [], lw=2)[0]  # initialize a line object

        self.target_plot = TargetPlot()

    def setup_prediction_plot(self, optimized_ensemble=None):

        def sum_oscillators(oscillators: List, n_oscillators: int) -> np.ndarray:
            return np.sum([oscillator.signal for oscillator in oscillators[:n_oscillators]], axis=0)

        @dataclass
        class PredictionPlot:
            fig = Figure()
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            if optimized_ensemble is not None and isinstance(optimized_ensemble, sample.Sample):
                meta_signal = meta_target.MetaTargetSmart(signal=optimized_ensemble.weighted_sum, duration=1, sampling_rate=SAMPLING_RATE)
            else:
                meta_signal = meta_target.MetaTargetSmart(signal=sum_oscillators(self.oscillator_plot.oscillators, self.oscillator_plot.n_oscillators),
                                                 duration=1, sampling_rate=SAMPLING_RATE)
            line = ax.plot([], [], lw=2)[0]

        self.prediction_plot = PredictionPlot()

    def setup_oscillator_plot(self, optimized_ensemble: sample.Sample = Union[sample.Sample | None]):
        @dataclass
        class OscillatorPlot:
            n_oscillators = int(self.n_oscillators_slider.value)
            n_rows, n_cols = data_analysis.infer_subplot_rows_cols(n_oscillators)
            fig, axs = plt.subplots(n_rows, n_cols)
            canvas = FigureCanvas(fig)

            oscillators = []
            lines = []
            if optimized_ensemble is not None and isinstance(optimized_ensemble, sample.Sample):
                for i in range(n_oscillators):
                    mt = meta_target.MetaTargetSmart(signal=optimized_ensemble.signal_matrix[i], duration=1, sampling_rate=SAMPLING_RATE)
                    oscillators.append(mt)
                oscillator_grid = oscillator_grid.OscillatorGrid(oscillators)
                for i in range(n_oscillators):
                    ax = data_analysis.select_axis(axs, n_cols, i)
                    lines.append(ax.plot([], [], lw=2)[0])
            else:            
                for _ in range(n_oscillators):
                    wave = self.select_wave_type(self.oscillator_wave_type.value,
                                                1, 1, 0, 0)
                    oscillators.append(wave)
                oscillator_grid = oscillator_grid.OscillatorGrid(oscillators)
                for i in range(n_oscillators):
                    ax = data_analysis.select_axis(axs, n_cols, i)
                    lines.append(ax.plot([], [], lw=2)[0])

        self.oscillator_plot = OscillatorPlot()

    def setup_target_webview(self):
        self.target_web_view = toga.WebView(style=Pack(flex=1))

    def setup_prediction_webview(self):
        self.prediction_web_view = toga.WebView(style=Pack(flex=1))

    def setup_oscillators_webview(self):
        self.oscillators_web_view = toga.WebView(style=Pack(flex=1))

    def compose_window(self):
        spacer1 = toga.Box(style=Pack(height=30))
        spacer2 = toga.Box(style=Pack(height=30))
        animations = toga.Box(children=[self.target_web_view,
                                        spacer1,
                                        self.oscillators_web_view,
                                        spacer2,
                                        self.prediction_web_view], style=Pack(direction=COLUMN, flex=2))
        
        animation_and_controls = toga.Box(children=[animations, self.controls_bar], style=Pack(direction=ROW))
        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = animation_and_controls
        self.main_window.show()

    def start_oscillator_animation(self):
        if not self.generating_oscillators:
            self.generating_oscillators = True
            self.oscillator_animation_thread = threading.Thread(target=self.generate_oscillator_gif)
            self.oscillator_animation_thread.start()

    def start_prediction_animation(self):
        if not self.generating_prediction:
            self.generating_prediction = True
            self.prediction_animation_thread = threading.Thread(target=self.generate_prediction_gif)
            self.prediction_animation_thread.start()

    def start_target_animation(self):
        if not self.generating_target:
            self.generating_target = True
            self.target_animation_thread = threading.Thread(target=self.generate_target_gif)
            self.target_animation_thread.start()

    def generate_target_gif(self):
        self.target_animation = animation.FuncAnimation(
            self.target_plot.fig,
            self.target_plot.target.animate,
            fargs=(NUM_FRAMES, self.target_plot.fig, self.target_plot.line,),
            frames=NUM_FRAMES,
            interval=20,
            repeat=True
        )
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as temp:
            self.target_gif = Path(temp.name)
        self.target_animation.save(self.target_gif, writer=PillowWriter(fps=80))
        self.generating_target = False

        # Update the WebView to display the new gif
        self.target_web_view.url = f"http://localhost:{PORT}/{self.target_gif.name}"

    def generate_prediction_gif(self):
        self.prediction_animation = animation.FuncAnimation(
            self.prediction_plot.fig,
            self.prediction_plot.meta_signal.animate,
            fargs=(NUM_FRAMES, self.prediction_plot.fig, self.prediction_plot.line,),
            frames=NUM_FRAMES,
            interval=20,
            repeat=True
        )
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as temp:
            self.prediction_gif = Path(temp.name)
        self.prediction_animation.save(self.prediction_gif, writer=PillowWriter(fps=80))
        self.generating_prediction = False

        # Update the WebView to display the new gif
        self.prediction_web_view.url = f"http://localhost:{PORT}/{self.prediction_gif.name}"

    def generate_oscillator_gif(self):
        # self.oscillator_animation.event_source.stop()
        self.oscillator_animation = animation.FuncAnimation(
            self.oscillator_plot.fig,
            self.oscillator_plot.oscillator_grid.animate,
            fargs=(NUM_FRAMES, self.oscillator_plot.fig, self.oscillator_plot.lines,),
            frames=NUM_FRAMES,
            interval=20,
            repeat=True
        )
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as temp:
            self.oscillator_gif = Path(temp.name)
        self.oscillator_animation.save(self.oscillator_gif, writer=PillowWriter(fps=80))
        self.generating_oscillators = False

        # Update the WebView to display the new gif
        self.oscillators_web_view.url = f"http://localhost:{PORT}/{self.oscillator_gif.name}"

    def start_server(self):
        # Change the current working directory to the directory of the temporary file
        os.chdir(self.placeholder_gif.parent)
        
        handler = http.server.SimpleHTTPRequestHandler
        self.httpd = socketserver.TCPServer(("", PORT), handler)
        self.server_thread = threading.Thread(target=self.httpd.serve_forever)
        self.server_thread.start()
        
        # Load the GIF file into the WebView
        self.target_web_view.url = f"http://localhost:{PORT}/{self.placeholder_gif.name}"
        self.prediction_web_view.url = f"http://localhost:{PORT}/{self.placeholder_gif.name}"
        self.oscillators_web_view.url = f"http://localhost:{PORT}/{self.placeholder_gif.name}"

def main():
    return oscillators()