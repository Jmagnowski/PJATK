# Filename: forest_reducer.py

from operator import itemgetter
import sys

# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()

    print('{}'.format(line))
