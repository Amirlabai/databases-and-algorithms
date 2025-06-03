################
#              #
#  Line Class  #
#              #
################

from graphics import *
from point import *
import math

class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def move(self,dx,dy):
        "Move line by dx and dy"
        self.p1.move(dx,dy)
        self.p2.move(dx,dy)

    def length(self):
        x1 = self.p1.x
        y1 = self.p1.y
        x2 = self.p2.x
        y2 = self.p2.y
        len = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        return len

    def middle(self):
        x = (self.p1.x + self.p2.x) / 2
        y = (self.p1.y + self.p2.y) / 2
        return Point(x,y)

    def draw(self, **kwargs):
        kwargs.setdefault('width', 2)
        kwargs.setdefault('fill', 'blue')
        x1 = self.p1.x
        y1 = self.p1.y
        x2 = self.p2.x
        y2 = self.p2.y
        id = canvas.create_line(x1, y1, x2, y2, **kwargs)
        self.id = id
        return id

    @classmethod
    def from_coords(cls, x1, y1, x2, y2):
        "Construct Line from 4 coordinates"
        return cls(Point(x1,y1), Point(x2,y2))

    # this is how a Line object will be print(ed with the Python print( statement:
    def __str__(self):
        return "Line(%s,%s)"  %  (str(self.p1) , str(self.p2))


#--------------------------------------------------

def test1():
    print( "----- Testing The Line Class -----")
    p1 = Point(100,100)
    p2 = Point(400,500)
    l = Line(p1,p2)
    print( "Testing the Python print( statement:")
    print( l)
    print( "Line length =", l.length())
    mid = "Line middle point:", l.middle()
    print( mid)
    dx = 50
    dy = -40
    print( "Now we move the line by dx=%d and dy=%d"  %  (dx,dy))
    l.move(dx,dy)
    print( l)
    print( "Test 1: PASSED")

def test2():
    p1 = Point(100,100)
    p2 = Point(400,500)
    l = Line(p1,p2)
    assert l.p1 == p1
    assert l.p2 == p2
    assert l.length() == 500
    l.move(50, -40)
    assert l.p1.x == 150 and l.p1.y == 60
    assert l.p2.x == 450 and l.p2.y == 460
    print( "Test 2: PASSED")

def test3():
    p1 = Point(100,100)
    p2 = Point(400,500)
    l = Line(p1,p2)
    print( l)
    l.draw()
    print( "Moving the line by dx=30, dy=-10")
    l.move(30,-10)
    print( l)
    print( "Redrawing the line")
    l.draw(fill="maroon", width=4, dash=(1,1))
    l.p1.draw()
    l.p2.draw()
    #mainloop()

if __name__ == "__main__":
    test1()
    test2()
    #test3()