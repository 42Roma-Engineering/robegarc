import sys

def my_min(x = 0, y = 0, z = 0):
    if (x < y) :
        min = x
    else :
        min = y
    if (min < z) :
        return(min)
    else :
        return(z)

def main() :
    if (len(sys.argv) != 4) :
        print("Error! Usage: python3 ft_min.py <x> <y> <z>")
    else :
        try :
            print("The min is:", my_min(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])))
        except Exception :
            print("Invalid data type")

if __name__ == "__main__":
    main()
