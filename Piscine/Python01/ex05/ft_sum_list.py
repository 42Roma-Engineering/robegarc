def sum_list(array) :
    sum = 0
    for i in array :
        sum += i
    return (sum)

def main() :
    array = []
    while True :
        x = int(input("Insert integer: "))
        array += [x]
        if (x == 0) :
            return (array)

if __name__ == "__main__":  
    try :
        array = main()
        print("The sum is:", sum_list(array))
    except Exception :
        print("Invalid data type")
