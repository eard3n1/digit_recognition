# digit_prediction
A simple CNN trained on MNIST with a Tkinter GUI for drawing and predicting digits.

## Features
- CPU-only for simplicity
- GUI canvas for drawing digits
- Real-time preprocessing for MNIST-like input
- Shows prediction and confidence

## Dependencies
- Python 3.10+
- torch
- torchvision
- Pillow
- Tkinter (built-in)

## Setup
1. Install requirements:
```bash
pip install -r requirements.txt
```
2. Train the model:
```bash
python train.py
```

## Options
- Use the GUI:
```bash
python gui.py
```
- Put an image named <code>digit.png</code> inside root directory & run:
```bash
python mnist/infer.py
```

## License
MIT License