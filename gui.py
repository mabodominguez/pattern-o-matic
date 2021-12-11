# import the necessary packages
import glob
from tkinter import *
from tkinter import Image
from tkinter import ttk
root = Tk()
root.title("Pattern!")
# root.iconbitmap("new_shape10rotated_image.png")
# root.geometry("1000x800")
CANVAS_PX_PER_IN = 35
in_dir = "pinafore_1_pattern_shapes/rotated_shapes"

fabric_width = 14
fabric_length = 81

# w = 30 * CANVAS_PX_PER_IN
# h = 72 * CANVAS_PX_PER_IN
w = fabric_width * CANVAS_PX_PER_IN
h = fabric_length * CANVAS_PX_PER_IN
x = int(w/4)
y = int(h/4)


class Fabric(Canvas):
    def __init__(self, parent, pattern_imgs, **kwargs):
        super().__init__(parent, **kwargs)
        self.bind("<Button-1>", self.select)
        self.bind("<B1-Motion>", self.move)
        self.current_i = 0
        self.pattern_imgs = []
        self.pattern_photos = []
        for img in pattern_imgs:
            image = PhotoImage(file=img)
            my_image = self.create_image(x, y, image=image)
            self.pattern_imgs.append(my_image)
            self.pattern_photos.append(image)
        (x0, y0, x1, y1) = self.bbox(self.pattern_imgs[self.current_i])
        self.current_rect = self.create_rectangle(
            x0, y0, x1, y1, outline='yellow')
        self.create_rectangle(
            20, 20, 20+(2*CANVAS_PX_PER_IN), 20+(2*CANVAS_PX_PER_IN), fill='red')
        self.create_text(
            40, 40, text="test square\nshould measure 2\"x 2\"", anchor='nw')

    # def xy(self, event):
    #     global lastx, lasty
    #     lastx, lasty = self.canvasx(event.x), self.canvasy(event.y)

    def select(self, event):
        self.current_i = (self.current_i + 1) % len(self.pattern_imgs)
        (x0, y0, x1, y1) = self.bbox(self.pattern_imgs[self.current_i])
        self.delete(self.current_rect)
        self.current_rect = self.create_rectangle(
            x0, y0, x1, y1, outline='yellow')

    def move(self, event):
        self.coords(self.pattern_imgs[self.current_i-1], self.canvasx(
            event.x), self.canvasy(event.y))
        x0, y0, x1, y1 = self.bbox(self.pattern_imgs[self.current_i-1])
        self.coords(self.current_rect, x0, y0, x1, y1)


# my_canvas = Canvas(root, width=w, height=h, bg="gray")
# my_canvas.pack(pady=20)
img_names = []

for img in glob.glob(f"{in_dir}/*.png"):
    img_names.append(img)


horz = ttk.Scrollbar(root, orient=HORIZONTAL)
vert = ttk.Scrollbar(root, orient=VERTICAL)
my_fabric = Fabric(root, img_names, bg='SlateGray1', scrollregion=(
    0, 0, h+10, w+10), yscrollcommand=vert.set, xscrollcommand=horz.set, width=w, height=h)
horz['command'] = my_fabric.xview
vert['command'] = my_fabric.yview


my_fabric.grid(column=0, row=0, sticky=(N, W, E, S))
horz.grid(column=0, row=1, sticky=(W, E))
vert.grid(column=1, row=0, sticky=(N, S))
my_label = Label(
    text="Click mouse to toggle between pattern pieces. Check first that all pattern pieces are there")
my_label.grid(column=0, row=0, sticky=(S, W))


root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)


root.mainloop()
