#################
#               #
#  Point Class  #
#               #
#################
print("in eda.point")
from graphics import *
print(rootWindow)


class Point:
    "Create an object of type Point from two integers"
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def move(self,dx,dy):
        "Move point by dx and dy"
        self.x += dx
        self.y += dy

    def draw(self, **kwargs):
        radius = kwargs.pop('radius', 4)
        kwargs.setdefault('width', 0)
        kwargs.setdefault('fill', 'black')
        kwargs.setdefault('tags', ['POINT'])
        x1 = self.x - radius
        y1 = self.y - radius
        x2 = self.x + radius
        y2 = self.y + radius
        id = canvas.create_rectangle(x1, y1, x2, y2, **kwargs)
        self.id = id
        return id

    def text(self, t, **kwargs):
        dx = kwargs.pop('dx', 0)
        dy = kwargs.pop('dy', -4)
        kwargs.setdefault('anchor', 's')
        kwargs.setdefault('font', 'Consolas 12')
        kwargs.setdefault('tags', ['TEXT', 'POINT'])
        kwargs['text'] = t
        id = canvas.create_text(self.x + dx, self.y + dy, **kwargs)
        return id

    # this is how a Point object will be printed with the Python print statement:
    def __str__(self):
        return "Point(%d,%d)"  %  (self.x, self.y)

#-----------------------------------------

def test1():
    print("\n===== Testing The Point Class =====")
    p1 = Point(20,20)
    p2 = Point(50,60)
    print("Testing the Python print statement on Point p1:")
    print( p1)
    print( "Testing the Python print statement on Point p2:")
    print( p2)
    print( "Test 1: PASSED")

def test2():
    p1 = Point(20,20)
    p2 = Point(50,60)
    assert p1.x == 20 and p1.y == 20
    assert p2.x == 50 and p2.y == 60
    p1.move(100, 200)
    p2.move(100, 200)
    assert p1.x == 120 and p1.y == 220
    assert p2.x == 150 and p2.y == 260
    print("Test 2: PASSED")

def test3():
    p1 = Point(20,30)
    p2 = Point(70,80)
    p3 = Point(130, 150)
    print(p1, p2, p3)
    print("Testing the Point draw method:")
    p1.draw()
    p2.draw(radius=8, fill="blue")
    p3.draw(fill="red", width=2)



if __name__ == "__main__":
    #test1()
    #test2()
    #test3()
    #mainloop()
    pass