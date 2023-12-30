a = input("Insert a string: ")
b = len(a)

if (b > 20) :
    print(a[len(a) - 20 : len(a)])
else :
    len_spaces = 20 - len(a)
    str_space = " " * len_spaces
    print(str_space, a)
