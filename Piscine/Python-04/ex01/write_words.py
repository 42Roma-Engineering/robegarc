def read_words(filename):
    words = []
    with open(filename, 'r') as file:
        for line in file:
            words.append(line)
    return words

try :
    char_len = int(input("Insert an integer: "))
    file_name = "words.txt"

    words = read_words(file_name)
    words.sort()

    print("The words longer than", char_len, "have been written on \"long_words.txt\"")
    with open("long_words.txt", "w+") as file :
        for elem in words :
            word = elem.strip("\n")
            length = len(word)
            ind = word.find(" ")
            if length >= char_len :
                if ind != -1 :
                    file.write(word[0:ind])
                else :
                    file.write(word)
                file.write("\n")
    file.close()
except Exception :
    print("Invalid data type")
