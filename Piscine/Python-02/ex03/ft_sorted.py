import sys


def check_order(array) :
    sort = sorted(array, reverse=True)
    for i in range(0, len(array)) :
        if array[i] != sort[i] :
            return False
    return True


if (len(sys.argv) < 3) :
    print("Error! You must insert at least 2 numbers")
else :
    try :
        i = [int(i) for i in sys.argv[1:]]
        if check_order(i) == True :
            print("The inserted sequence is sorted!")
        else :
            print("The inserted sequence is not sorted!")
            print("The correct order is", sorted(i, reverse=True))
    except Exception :
        print("Invalid data type")
