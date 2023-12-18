import sys

if (len(sys.argv) != 3) :
    print("Number of arguments not valid")
else :
    try :
        arg1 = sys.argv[1]
        arg2 = sys.argv[2]

        arg1_float = float(arg1)
        arg2_float = float(arg2)

        if (arg1_float > arg2_float) :
            print(arg1_float, "is greater than", arg2_float)
        elif (arg1_float < arg2_float) :
            print(arg1_float, "is less than", arg2_float)
        else:
            print(arg1_float, "is equal to", arg2_float)
    except Exception :
        print("Invalid data type")
