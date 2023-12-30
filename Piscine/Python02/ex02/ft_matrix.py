import sys


def fill_mtx(rows, columns) :
    mtx = [[column for column in range(columns)] for row in range(rows)]
    i = 0
    while i < rows :
        j = 0
        while j < columns :
            v = float(input(f"Insert the element in position ({i}, {j}): "))
            mtx[i][j] = v
            j += 1
        i += 1
    return (mtx)


def print_mtx(mtx, rows) :
    print("The inserted matrix is:")
    i = 0
    while i < rows :
        print(mtx[i])
        i += 1


def mtx_sum_rows(mtx) :
    sum_array = []
    sum = 0
    for row in mtx:
        for element in row:
            sum += element
        sum_array.append(sum)
        sum = 0
    print("The sum over rows is:")
    print(sum_array)


def mtx_sum_cols(mtx, rows, columns) :
    sum_array = []
    sum = 0
    i = 0
    while i < columns :
        j = 0
        while j < rows :
            sum += mtx[j][i]
            j += 1
        sum_array.append(sum)
        sum = 0
        i += 1
    print("The sum over columns is:")
    print(sum_array)



if (len(sys.argv) != 3) :
    print("Error! Usage: python3 ft_matrix.py <n> <m>")
else :
    try :
        rows = int(sys.argv[1])
        columns = int(sys.argv[2])
        mtx = fill_mtx(rows, columns)
        print_mtx(mtx, rows)
        mtx_sum_rows(mtx)
        mtx_sum_cols(mtx, rows, columns)
    except Exception :
        print("Invalid data type")
