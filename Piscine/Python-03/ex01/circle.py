class Point :

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self) :
        return f"({self.x}, {self.y})"


class Circle(Point) :

    def __init__(self, p, radius: int) :
        self.p = p
        self.radius = radius

    def __str__(self) :
        return f"Circle of center {self.p} and radius {self.radius}"

def main() :
    lst = []
    lst.append(150)
    lst.append(100)

    p = Point(int(150), int(100))
    c = Circle(p, 75)
    print(c)

if __name__ == "__main__" :
    main()
