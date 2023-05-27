import sys

INPUT = sys.argv[1]
OUTPUT = sys.argv[2]
BASE_AMOUNT = int(sys.argv[3])

fens_in = open(INPUT, 'r')
fens_out = open(OUTPUT, 'w')

for i in range(3, 33, 1):
    j = 0
    c = 0
    while True:
        line = fens_in.readline()

        if not line:
            break
        s = line.split(" ")
        c = int(s[0])
        if c == i:
            fens_out.write(s[1])
            j += 1
        if j >= BASE_AMOUNT:
            break
        if c > i:
            break

    print(f"got {j} samples of {i} pieces")

fens_in.close()
fens_out.close()
