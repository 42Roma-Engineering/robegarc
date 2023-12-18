import sys

def find_max(x, y):
    if (x > y) :
        return (x)
    else :
        return (y)

if (len(sys.argv) != 4) :
    print("Error! Usage: python3 ft_max.py <x> <y> <z>")
else :
    try :
        max = find_max(float(sys.argv[1]), float(sys.argv[2]))
        max = find_max(max, float(sys.argv[3]))
        print("The max is:", max)
    except Exception :
        print("Invalid data type")
