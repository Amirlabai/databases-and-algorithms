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
    rootWindow
except NameError:
    print("Setting up TK window")
    rootWindow = tkinter.Tk()
    rootWindow.wm_title("main window")
    rootWindow.resizable(width=True, height=True)
    rootFrame = tkinter.Frame(rootWindow, bg="white")
    rootFrame.pack(fill=tkinter.BOTH, expand=True)

    # יצירת סרגל גלילה אופקי
    x_scrollbar = tkinter.Scrollbar(rootFrame, orient=tkinter.HORIZONTAL)
    x_scrollbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)

    # יצירת סרגל גלילה אנכי
    y_scrollbar = tkinter.Scrollbar(rootFrame, orient=tkinter.VERTICAL)
    y_scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)

    # יצירת הקנבס עם תמיכה בגלילה
    canvas = tkinter.Canvas(
        rootFrame,
        bg="white",
        scrollregion=(0, 0, 10000, 20000),  # ← אפשר לשנות לפי גודל העץ שלך
        xscrollcommand=x_scrollbar.set,
        yscrollcommand=y_scrollbar.set
    )
    canvas.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=True)

    # קישור בין הסקרולרים לקנבס
    x_scrollbar.config(command=canvas.xview)
    y_scrollbar.config(command=canvas.yview)

    def on_mouse_wheel(event):
        """Callback for vertical mouse wheel scrolling."""
        canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def on_shift_mouse_wheel(event):
        """Callback for horizontal mouse wheel scrolling with Shift key."""
        canvas.xview_scroll(-1 * (event.delta // 120), "units")

    # Bind the mouse wheel for vertical scrolling
    canvas.bind("<MouseWheel>", on_mouse_wheel)

    # Bind Shift + mouse wheel for horizontal scrolling
    canvas.bind("<Shift-MouseWheel>", on_shift_mouse_wheel)


def show_canvas():
    if not canvas.winfo_ismapped():
        tkinter.mainloop()

def update_canvas():
    rootWindow.update()

colors = 10 * ['red1', 'DarkGoldenrod2', 'yellow2', 'DarkOliveGreen1', 'chartreuse1', 'green3', 'DarkSlateGray3', 'MediumPurple3', 'MediumOrchid4', 'MediumOrchid3', 'thistle2', 'gray']
color_index = -1

def get_next_color():
    global color_index
    global colors
    color_index += 1
    if color_index >= len(colors):
        color_index = 0
    return colors[color_index]