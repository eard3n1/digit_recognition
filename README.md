# digit_recognition
A simple <b>CNN</b> built with <a href="https://pytorch.org/">PyTorch</a> trained on <b>MNIST</b>.

## Features
- CPU-only for simplicity
- GUI canvas for drawing digits
- Real-time preprocessing for MNIST-like input
- Graph for loss & accuracy during training process

## Dependencies
- Python 3.9+ (3.10)
- Torch
- Torchvision
- Pillow
- Matplotlib
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
python architecture/infer.py
```

## License
MIT License
