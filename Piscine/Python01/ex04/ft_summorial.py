import sys

if len(sys.argv) != 2 :
    print("Error! Usage: python3 ft_summorial.py <n>")
    exit()
try :
    if int(sys.argv[1]) < 0 :
        print("Error! n must be >= 0")
    elif int(sys.argv[1]) == 0 :
        print("The sum is: 0")
    else :
        v = int(sys.argv[1])
        r = 0
        for i in range(1, v + 1) :
            r += i
        print("The sum is:", r)
except Exception :
    print("Invalid data type")
