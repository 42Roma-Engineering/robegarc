def absolute_expression_solver(x):
    y = float(eval(x))
    if (y < 0) :
        y *= -1
    return (y)

x = input("Insert an expression: ")
try :
    r = absolute_expression_solver(x)
    print("The result is:", r)
except Exception :
    print("Invalid data type")
