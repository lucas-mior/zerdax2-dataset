import sys

INPUT = sys.argv[1]
OUTPUT = sys.argv[2]

fens_in = open(INPUT, 'r')
fens_out = open(OUTPUT, 'w')

for i in range(3, 33, 1):
    j = 0
    while True:
        line = fens_in.readline()
        count = 0

        if not line:
            break
        for c in line:
            if c.isalpha():
                count += 1
        a = f"{count} {line}"
        fens_out.write(a)

fens_in.close()
fens_out.close()
