from pathlib import Path
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import numpy as np
import threading
import http.server
import socketserver
import os
import tempfile

PORT = 8005

class oscillators(toga.App):
    def startup(self):
        self.on_exit = self.exit_handler

        self.setup_plot()
        self.setup_webview()
        self.setup_window()
        self.start_animation()
        self.start_server()
    
    async def exit_handler(self, app):
        # Return True if app should close, and False if it should remain open
        if await self.main_window.confirm_dialog(
            "Toga", "Are you sure you want to quit?"
        ):
            self.httpd.server_close() # free the port
            self.httpd.shutdown() # stop server forever
            self.server_thread.join() # wait for the server to stop
            os.remove(self.gif_file) # remove the temporary file

            return True
        else:
            return False

    def setup_plot(self):
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.x = np.linspace(0, 2*np.pi, 100)
        self.line, = self.ax.plot(self.x, np.sin(self.x))

    def setup_webview(self):
        self.web_view = toga.WebView(style=Pack(flex=1))

    def setup_window(self):
        main_box = toga.Box(children=[self.web_view])
        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = main_box
        self.main_window.show()

    def start_animation(self):
        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=100, interval=20, blit=True, repeat=True)
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as temp:
            self.gif_file = Path(temp.name)
        self.ani.save(self.gif_file, writer=PillowWriter(fps=30))

    def start_server(self):
        # Change the current working directory to the directory of the temporary file
        os.chdir(self.gif_file.parent)
        
        handler = http.server.SimpleHTTPRequestHandler
        self.httpd = socketserver.TCPServer(("", PORT), handler)
        self.server_thread = threading.Thread(target=self.httpd.serve_forever)
        self.server_thread.start()
        
        # Load the GIF file into the WebView
        self.web_view.url = f"http://localhost:{PORT}/{self.gif_file.name}"

    def animate(self, i):
        self.line.set_ydata(np.sin(self.x + i / 50))
        self.canvas.draw()
        return self.line,

def main():
    return oscillators()