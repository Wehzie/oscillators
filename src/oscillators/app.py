import asyncio
from pathlib import Path
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
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
from scipy.signal import sawtooth, chirp
import threading
from typing import Final

PORT = 8002
NUM_FRAMES = 100
RNG: Final = np.random.default_rng(5)

class oscillators(toga.App):
    def startup(self):
        self.on_exit = self.exit_handler

        self.setup_webview()
        self.setup_window()
        self.setup_controls()

        self.setup_plot()

        self.create_placeholder_gif()
        
        self.start_animation()
        self.start_server()

    def create_placeholder_gif(self):
        # Create a placeholder GIF file
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as temp:
            self.gif_file = Path(temp.name)
        img = PIL_Image.new('RGB', (800, 400), color = (255, 255, 255))  # White background
        d = ImageDraw.Draw(img)
        font_size = 60
        fnt = ImageFont.load_default().font_variant(size=font_size)
        text = "Generating animation."
        # estimate the width and height of the text from number of characters

        text_width, text_height = 1/2 * font_size * len(text), 1/2 * font_size
        x = (img.width - text_width) / 2
        y = (img.height - text_height) / 2
        d.text((x, y), text, font=fnt, fill=(0, 0, 0))
        img.save(self.gif_file, 'GIF')
    
    async def exit_handler(self, app):
        # Return True if app should close, and False if it should remain open
        if await self.main_window.confirm_dialog(
            "Toga", "Are you sure you want to quit?"
        ):
            self.httpd.server_close() # free the port
            self.httpd.shutdown() # stop server forever
            self.server_thread.join() # wait for the server to stop
            if self.animation_thread.is_alive():
                self.animation_thread.join()
            os.remove(self.gif_file) # remove the temporary file

            return True
        else:
            return False

    def setup_controls(self):
        self.wave_type = toga.Selection(
            items=['Sine', 'Triangle', 'Square', 'Sawtooth', 'Chirp', 'Uniform Noise'],
            style=Pack(padding=10),
            on_change=self.on_wave_type_select
        )
        pad_left_right = Pack(padding=(0, 10, 0, 10))
        self.amplitude_slider = toga.Slider(min=0.1, max=10, value=1, on_release=self.on_slider_change, style=pad_left_right)
        self.offset_slider = toga.Slider(min=-5, max=5, value=0, on_release=self.on_slider_change, style=pad_left_right)
        self.phase_slider = toga.Slider(min=0, max=2*np.pi, value=0, on_release=self.on_slider_change, style=pad_left_right)
        self.frequency_slider = toga.Slider(min=0.1, max=10, value=1, on_release=self.on_slider_change, style=pad_left_right)

        controls_box = toga.Box(
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

        self.main_window.content.add(controls_box)

    def on_slider_change(self, widget):
        pass

    def on_wave_type_select(self, widget):
        # Stop the current animation
        self.ani.event_source.stop()

        # Start a new animation with the selected wave type
        self.start_animation()

        # Update the WebView to display the new gif
        self.web_view.url = f"http://localhost:{PORT}/{self.gif_file.name}"

    def setup_plot(self):
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.x = np.linspace(0, 2*np.pi, 500)
        self.line, = self.ax.plot(self.x, np.sin(self.x))

    def setup_webview(self):
        self.web_view = toga.WebView(style=Pack(flex=1))

    def setup_window(self):
        main_box = toga.Box(children=[self.web_view])
        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = main_box
        self.main_window.show()

    def start_animation(self):
        # Create a new thread to generate the GIF
        self.animation_thread = threading.Thread(target=self.generate_gif)
        self.animation_thread.start()

    def generate_gif(self):
        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=NUM_FRAMES, interval=20, blit=True, repeat=True)
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as temp:
            self.gif_file = Path(temp.name)
        self.ani.save(self.gif_file, writer=PillowWriter(fps=30))

        # Update the WebView to display the new gif
        self.web_view.url = f"http://localhost:{PORT}/{self.gif_file.name}"

    def start_server(self):
        # Change the current working directory to the directory of the temporary file
        os.chdir(self.gif_file.parent)
        
        handler = http.server.SimpleHTTPRequestHandler
        self.httpd = socketserver.TCPServer(("", PORT), handler)
        self.server_thread = threading.Thread(target=self.httpd.serve_forever)
        self.server_thread.start()
        
        # Load the GIF file into the WebView
        self.web_view.url = f"http://localhost:{PORT}/{self.gif_file.name}"

    def animate(self, frame_index):
        # Calculate the phase shift
        phase_shift = (frame_index / NUM_FRAMES) * 2 * np.pi

        # Generate the wave with the phase shift
        if self.wave_type.value == 'Sine':
            self.line.set_ydata(np.sin(self.x + phase_shift))
        elif self.wave_type.value == 'Triangle':
            self.line.set_ydata(sawtooth(self.x + phase_shift, 0.5))
        elif self.wave_type.value == "Square":
            self.line.set_ydata(np.sign(np.sin(self.x + phase_shift)))
        elif self.wave_type.value == "Sawtooth":
            self.line.set_ydata(sawtooth(self.x + phase_shift, 1))
        elif self.wave_type.value == "Chirp":
            x_max = self.x[-1]
            phase_shift_vary_up = (frame_index / NUM_FRAMES) * x_max
            phase_shift_vary_down = x_max / (1+(frame_index / NUM_FRAMES))
            single_chirp_up = chirp(self.x + phase_shift_vary_up, f0=1, f1=3, t1=x_max, method='linear')
            single_chirp_down = chirp(self.x - phase_shift_vary_down, f0=3, f1=1, t1=x_max, method='linear')
            if frame_index < NUM_FRAMES / 2:
                full_chirp = single_chirp_up
            else:
                full_chirp = single_chirp_down
            self.line.set_ydata(full_chirp)
        elif self.wave_type.value == "Uniform Noise":
            noise = RNG.uniform(-1, 1, self.x.size)
            self.line.set_ydata(noise)

        self.canvas.draw()
        return self.line,

def main():
    return oscillators()