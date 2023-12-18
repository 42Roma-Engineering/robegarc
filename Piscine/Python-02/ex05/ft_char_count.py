import sys

if len(sys.argv) != 2 :
    print("Error! No string given")
else :
    inp = sys.argv[1]
    freq = {}

    for elem in inp:
        if elem in freq: 
            freq[elem] += 1
        else: 
            freq[elem] = 1

    k = freq.keys()
    sorted_k = sorted(k)

    d = {}
    for key in sorted_k :
        d[key] = freq[key]

    print("Char count:")
    for key, value in d.items() :
        print(key, "=", value)
