try :
    a = int(input("Insert hours: "))
    b = int(input("Insert minutes: "))
    c = int(input("Insert seconds: "))

    print("Total seconds:", (a * 60 * 60) + (b * 60) + c)
except Exception : 
    print("Please submit only integers")
