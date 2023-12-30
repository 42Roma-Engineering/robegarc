a = input("Insert a string: ")
try :
    b = int(input("Insert an interger: "))

    if b > len(a) or b < 0 :
        print("Error: index out of range")
    elif b == 0 :
        print(a[b])
    else :
        letter = a[b]
        next_to_last_ind = b - 1
        next_to_last_letter = a[next_to_last_ind]
        print(letter, next_to_last_letter)
except Exception :
    print("Invalid data type")
