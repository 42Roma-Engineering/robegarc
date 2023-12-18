import sys

class Point :

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self) :
        return f"({self.x}, {self.y})"


class Circle :

    def __init__(self, p, radius) :
        self.p = p
        self.radius = radius

    def contains(self, p) :
        dis = (((p.x - 0) ** 2 + (p.y - 0) ** 2) ** 0.5)
        if dis <= self.radius:
            return True
        return False

    def __str__(self) :
        return "Circle of center(%s, %s) and radius %s" % (self.p.lst[0], self.p.lst[1], self.radius)


def main() :
    if len(sys.argv) != 3 :
        print("Error! Wrong number of arguments")
    else :
        try :
            p = Point(float(sys.argv[1]), float(sys.argv[2]))
            p_zero = Point(0, 0)
            c = Circle(p, 1)
            if c.contains(p) == True :
                print("Point", p, "lies in the Circle of center", p_zero, "and radius", c.radius)
            else :
                print("Point", p, "lies out the Circle of center", p_zero, "and radius", c.radius)
        except ValueError :
            print("Error! Please provide valid numerical arguments")


if __name__ == "__main__":
    main()
