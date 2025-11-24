import tkinter as tk
from PIL import Image, ImageDraw
from mnist.infer import load_model, predict

model = load_model()

class GuiApp:
    def __init__(self, master):
        self.master = master

        self.canvas_size = 400
        self.canvas = tk.Canvas(master, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()

        self.label = tk.Label(master, text="Draw a digit", font=("Helvetica", 16))
        self.label.pack()

        self.button_predict = tk.Button(master, text="Predict", command=self.on_predict, width=30, height=2)
        self.button_predict.pack()

        self.button_clear = tk.Button(master, text="Clear", command=self.clear_canvas, width=30, height=2)
        self.button_clear.pack()

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.center_window()

    def center_window(self):
        self.master.update_idletasks()
        width = self.master.winfo_width()
        height = self.master.winfo_height()
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.master.geometry(f"{width}x{height}+{x}+{y}")

    def paint(self, event):
        x, y = event.x, event.y
        r = 12
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill=255)
        self.label.config(text="Draw a digit")

    def on_predict(self):
        assumption, confidence = predict(model, self.image)
        self.label.config(text=f"Assumption: {assumption} | Confidence: ({confidence:.2f}%)")

if __name__ == "__main__":
    root = tk.Tk()
    app = GuiApp(root)
    root.mainloop()