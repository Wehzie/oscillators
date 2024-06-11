# Signal Approximation by a Sum of Oscillators - with Graphics!

![GUI demo video](demo.gif)

This repo introduces a graphical frontend (GUI) to a [project](https://github.com/Wehzie/master-thesis) developed as a master thesis.
The original projects addresses the question of function approximation by a sum of constant frequency oscillators.
Such oscillators can be realized in physical circuits, which can be more efficient for certain computations compared to digital compute.
The approach resembles Fourier analysis or synthesis.

Why are oscillators cool?
They can store some signals more efficiently than digital hardware.
This efficiency also applies to high frequency signals.
Such signals can control fast motors, for example in selective laser sintering (SLS).
Or think about electric candles.
Instead of a Turing-complete microcontroller controlling the flickering, how about a few oscillators, might be much cheaper.

A GUI allows for intuitive experimentation with oscillators.
Further building the UI allows me to try [BeeWare](https://beeware.org/).


## Running

Clone the project.

    git clone your/desired/path

Create a virtual environment in side the cloned project.

    python -m venv venv

Activate the virtual environment.

    source venv/bin/activate

Install dependencies.

    pip install -r requirements.txt

Start the app.

    briefcase dev

## Debugging

Run individual files.

```
# Example
python -m src.oscillators.optimization.gen_signal_python
```

## Feature ideas

- Pass selected wave function and distribution parameters from GUI to backend
- Upload audio files for real world signals
- Audio playback of wave functions
- Display Root Mean Squared Error (RMSE)