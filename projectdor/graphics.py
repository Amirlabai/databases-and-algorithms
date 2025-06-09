# we need a root window object and then a Canvas window object (child)
# in order to be able to draw points and lines.
# The Point and Line draw() method will use this canvas.
#
# A Canvas Window on which we can draw points, lines, rectangles, etc.
# See the Tkinter module, Canvas class, for more details.
# PyScripter may not handle Tkinter well, so to run this example,
# use the command line:
#                       python prog.py
import tkinter

try:
    rootWindow # type: ignore
except NameError:
    print("Setting up TK window")
    rootWindow = tkinter.Tk()
    rootWindow.wm_title("main window")
    rootWindow.resizable(width=True, height=True)
    rootFrame = tkinter.Frame(rootWindow, width=1820, height=970, bg="white")
    rootFrame.pack()
    canvas = tkinter.Canvas(rootFrame, width=1800, height=970, bg="white")
    canvas.pack()


def show_canvas():
    if not canvas.winfo_ismapped():
        tkinter.mainloop()

def update_canvas():
    rootWindow.update()

colors = 10 * ['red1', 'DarkGoldenrod2', 'yellow2', 'DarkOliveGreen1', 'chartreuse1', 'green3', 'DarkSlateGray3', 'MediumPurple3', 'MediumOrchid4', 'MediumOrchid3', 'thistle2', 'gray']
color_index = -1

def get_next_color(increment=0):
    global color_index
    global colors
    if increment > 0:
        return colors[increment]
    color_index += 1
    if color_index >= len(colors):
        color_index = 0
    return colors[color_index]

def grow_pixel(pixels=1):
    pixels = pixels*0.8
    return pixels