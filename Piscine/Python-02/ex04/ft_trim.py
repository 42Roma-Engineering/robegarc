import sys

def trim(array) :
    array.pop(0)
    array.pop(-1)


def main() :
    if (len(sys.argv) < 3) :
        print("Error! You must insert at least 2 values")
    else :
        i = sys.argv[1:]
        trim(i)
        print(f"The new list is: {i}")


if __name__ == "__main__":
    main()
