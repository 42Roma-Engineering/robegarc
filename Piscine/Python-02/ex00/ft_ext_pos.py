import sys

def print_max(array) :
    max = int(array[0])
    index = -1
    x = 0
    while x < len(array) :
        v = int(array[x])
        if (max < v) :
            max = v
            index = x
        x += 1
    print("The max is", max, "and its position is", index)

def print_min(array) :
    min = int(array[0])
    index = -1
    x = 0
    while x < len(array) :
        v = int(array[x])
        if (min > v) :
            min = v
            index = x
        x += 1
    print("The min is", min, "and its position is", index)


if (len(sys.argv) == 1) :
    print("Error! Usage: python3 ft_ext_pos.py <x1> ... <xn>")
else :
    try :
        A = sys.argv[1:len(sys.argv)]
        print_min(A)
        print_max(A)
    except Exception :
        print("Invalid data type")
