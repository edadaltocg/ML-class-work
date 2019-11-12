"""A simple reader for data file for TSP problem"""


import sys

with open(sys.argv[1]) as data_file:
    nb = 0

    # first line is total number of vertices in file
    print("expected number of vertices : {0}\n".format(int(data_file.readline())))

    for line in data_file:
        print("reading point {0}: {1}".format(nb,
                                              tuple(float(x) for x in line.split())))
        nb = nb + 1

    print("\nnumber of vertices: {0}".format(nb))
