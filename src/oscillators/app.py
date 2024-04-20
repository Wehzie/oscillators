from pathlib import Path
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
import threading
import http.server
import socketserver
import os
import tempfile
import threading
from dataclasses import dataclass

import oscillators.target as target
import oscillators.plot_layout as plot_layout

PORT = 8001
NUM_FRAMES = 40
SAMPLING_RATE = 100


class oscillators(toga.App):
    def startup(self):
        self.generating_target = False
        self.generating_oscillators = False
        self.on_exit = self.exit_handler

        self.setup_target_webview()
        self.setup_oscillators_webview()
        self.setup_controls_bar()
        self.compose_window()

        self.setup_target_plot()
        self.setup_oscillator_plot()

        self.init_gifs()
        
        self.start_target_animation()
        self.start_oscillator_animation(None)
        self.start_server()

        self.global_counter = 0


    def init_gifs(self):
        self.target_gif = None
        self.oscillator_gif = None
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
                                                on_release=self.update_oscillator_parameters)
        self.oscillator_controls = toga.Box(
            children=[
                toga.Label('Number of oscillators:', style=Pack(padding=(10, 10, 0, 10))),
                self.n_oscillators_slider,
            ],
            style=Pack(direction=COLUMN)
        )

    def setup_target_controls(self):
        self.wave_type = toga.Selection(
            items=['Sine', 'Triangle', 'Square', 'Sawtooth', 'Inverse Sawtooth', 'Chirp', 'Beat', 'Damp Chirp',
                   'Smooth Gaussian Noise', 'Smooth Uniform Noise', 'Gaussian Noise', 'Uniform Noise'],
            style=Pack(padding=10),
            on_change=self.update_target_parameters
        )
        pad_left_right = Pack(padding=(0, 10, 0, 10))
        self.amplitude_slider = toga.Slider(min=0.1, max=10, value=1, on_release=self.update_target_parameters, style=pad_left_right)
        self.offset_slider = toga.Slider(min=-5, max=5, value=0, on_release=self.update_target_parameters, style=pad_left_right)
        self.phase_slider = toga.Slider(min=0, max=2*np.pi, value=0, on_release=self.update_target_parameters, style=pad_left_right)
        self.frequency_slider = toga.Slider(min=0.1, max=10, value=1, on_release=self.update_target_parameters, style=pad_left_right)

        self.target_controls = toga.Box(
            children=[
                self.wave_type,
                toga.Label('Amplitude:', style=Pack(padding=(10, 10, 0, 10))),
                self.amplitude_slider,
                toga.Label('Offset:', style=Pack(padding=(10, 10, 0, 10))),
                self.offset_slider,
                toga.Label('Phase:', style=Pack(padding=(10, 10, 0, 10))),
                self.phase_slider,
                toga.Label('Frequency:', style=Pack(padding=(10, 10, 0, 10))),
                self.frequency_slider
            ],
            style=Pack(direction=COLUMN)
        )

    def update_target_parameters(self, widget):
        # Stop the current animation
        self.target_animation.event_source.stop()

        # Initialize new target
        if self.wave_type.value == 'Sine':
            self.target_plot.target = target.SineTarget(1, SAMPLING_RATE, freq=self.frequency_slider.value, amplitude=self.amplitude_slider.value, offset=self.offset_slider.value, phase=self.phase_slider.value)
        elif self.wave_type.value == 'Triangle':
            self.target_plot.target = target.TriangleTarget(1, SAMPLING_RATE, freq=self.frequency_slider.value, amplitude=self.amplitude_slider.value, offset=self.offset_slider.value, phase=self.phase_slider.value)
        elif self.wave_type.value == "Square":
            self.target_plot.target = target.SquareTarget(1, SAMPLING_RATE, freq=self.frequency_slider.value, amplitude=self.amplitude_slider.value, offset=self.offset_slider.value, phase=self.phase_slider.value)
        elif self.wave_type.value == "Sawtooth":
            self.target_plot.target = target.SawtoothTarget(1, SAMPLING_RATE, freq=self.frequency_slider.value, amplitude=self.amplitude_slider.value, offset=self.offset_slider.value, phase=self.phase_slider.value)
        elif self.wave_type.value == "Inverse Sawtooth":
            self.target_plot.target = target.InverseSawtoothTarget(1, SAMPLING_RATE, freq=self.frequency_slider.value, amplitude=self.amplitude_slider.value, offset=self.offset_slider.value, phase=self.phase_slider.value)
        elif self.wave_type.value == "Chirp":
            self.target_plot.target = target.ChirpTarget(1, SAMPLING_RATE, stop_freq=self.frequency_slider.value, amplitude=self.amplitude_slider.value, offset=self.offset_slider.value) # TODO: phase
        elif self.wave_type.value == "Beat":
            self.target_plot.target = target.BeatTarget(1, SAMPLING_RATE, base_freq=self.frequency_slider.value, amplitude=self.amplitude_slider.value, offset=self.offset_slider.value, phase=self.phase_slider.value)
        elif self.wave_type.value == "Damp Chirp":
            self.target_plot.target = target.DampChirpTarget(1, SAMPLING_RATE, stop_freq=self.frequency_slider.value, amplitude=self.amplitude_slider.value, offset=self.offset_slider.value) # TODO: phase
        elif self.wave_type.value == "Smooth Gaussian Noise":
            self.target_plot.target = target.SmoothGaussianNoiseTarget(1, SAMPLING_RATE, amplitude=self.amplitude_slider.value, offset=self.offset_slider.value) # TODO: window-length
        elif self.wave_type.value == "Smooth Uniform Noise":
            self.target_plot.target = target.SmoothUniformNoiseTarget(1, SAMPLING_RATE, amplitude=self.amplitude_slider.value, offset=self.offset_slider.value) # TODO: window-length
        elif self.wave_type.value == "Gaussian Noise":
            self.target_plot.target = target.GaussianNoiseTarget(1, SAMPLING_RATE, amplitude=self.amplitude_slider.value, offset=self.offset_slider.value)
        elif self.wave_type.value == "Uniform Noise":
            self.target_plot.target = target.UniformNoiseTarget(1, SAMPLING_RATE, amplitude=self.amplitude_slider.value, offset=self.offset_slider.value)

        # Start a new animation with the selected wave type
        self.start_target_animation()

    def update_oscillator_parameters(self, widget):
        self.oscillator_plot = None
        self.setup_oscillator_plot()
        self.start_oscillator_animation(None)

    def setup_target_plot(self):
        @dataclass
        class TargetPlot:
            fig = Figure()
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            target = target.SineTarget(1, SAMPLING_RATE)
            line = ax.plot([], [], lw=2)[0]  # initialize a line object

        self.target_plot = TargetPlot()

    def setup_oscillator_plot(self):
        @dataclass
        class OscillatorPlot:
            n_oscillators = int(self.n_oscillators_slider.value)
            n_rows, n_cols = plot_layout.infer_subplot_rows_cols(n_oscillators)
            fig, axs = plt.subplots(n_rows, n_cols)
            canvas = FigureCanvas(fig)
            
            oscillators = [target.SineTarget(1, SAMPLING_RATE) for _ in range(n_oscillators)]
            oscillator_grid = target.OscillatorGrid(oscillators)
            lines = []
            for i in range(n_oscillators):
                ax = plot_layout.select_axis(axs, n_cols, i)
                lines.append(ax.plot([], [], lw=2)[0])

        self.oscillator_plot = OscillatorPlot()

    def setup_target_webview(self):
        self.target_web_view = toga.WebView(style=Pack(flex=1))

    def setup_oscillators_webview(self):
        self.oscillators_web_view = toga.WebView(style=Pack(flex=1))

    def compose_window(self):
        spacer = toga.Box(style=Pack(height=30))
        animations = toga.Box(children=[self.target_web_view, spacer, self.oscillators_web_view], style=Pack(direction=COLUMN, flex=2))
        
        animation_and_controls = toga.Box(children=[animations, self.controls_bar], style=Pack(direction=ROW))
        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = animation_and_controls
        self.main_window.show()

    def start_oscillator_animation(self, widget):
        if not self.generating_oscillators:
            self.generating_oscillators = True
            self.oscillator_animation_thread = threading.Thread(target=self.generate_oscillator_gif)
            self.oscillator_animation_thread.start()

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
        self.oscillators_web_view.url = f"http://localhost:{PORT}/{self.placeholder_gif.name}"

def main():
    return oscillators()