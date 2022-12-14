#!/usr/bin/awk -f
BEGIN {
    CONVERT[0] = 3
    CONVERT[1] = 0
    CONVERT[2] = 4
    CONVERT[3] = 5
    CONVERT[4] = 1
    CONVERT[5] = 2
    CONVERT[6] = 3
    CONVERT[7] = 0
    CONVERT[8] = 4
    CONVERT[9] = 5
    CONVERT[10] = 1
    CONVERT[11] = 2
}
{
    num = $1
    $1 = ""
    printf("%d%s\n", CONVERT[num], $0)
}
