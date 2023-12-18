import pickle

def count_word_len(filename):
    word_lengths = {}
    with open(filename, 'r') as file:
        for line in file:
            word = line.strip()
            length = len(word)
            if length in word_lengths:
                word_lengths[length] += 1
            else:
                word_lengths[length] = 1
    file.close()
    return word_lengths

file_name = "words.txt"
word_count_dict = count_word_len(file_name)
sorted_dict = dict(sorted(word_count_dict.items()))

with open("word_count.pickle", "wb") as pickle_file:
    pickle.dump(sorted_dict, pickle_file)
pickle_file.close()

with open("word_count.pickle", "rb") as pickle_file:
    loaded_dict = pickle.load(pickle_file)
pickle_file.close()

""""""""""
Object Serialization - backup (?)
It is the process of storing a data structure in memory
so that you can load or transmit it
when required without losing its current state

dictionary from ocount_word_len():
keys ===> word lengths 
values ===> count of words with that length

"wb" ===> writing in binary
Serialize this dictionary with pickle.dump()
and save it to a binary file "word_count.pickle"
The purpose of "serialize"
is to preserve the dictionary's structure and content
in a serialized format that can be easily loaded back into Python later.

"rb" ===> read in binary
Loading and Displaying the Pickled Data:
After saving the data into "word_count.pickle", 
the code then reads the pickled file using pickle.load()
pickle.load() can read the serialized binary file "word_count.pickle"
The loaded content is stored in "loaded_dict"
"""""""""""
