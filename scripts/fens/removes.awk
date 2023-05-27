#!/usr/bin/awk -f

BEGIN {
    FS="."
}

{
    last = gensub(".+ ([0-9]*)", "\\1", "g", $(NF-1))
    last = strtonum(last)

    move = int(rand() * last + 1)
    move += 2
    if (move > last)
        move = last
    if (move < 4)
        move += 4
    if (move > last)
        move = last
    # if (last >= 100) {
    #     move = 100
        gsub(move ".+$", "", $0)
    # }
    print $0
}
