#!/usr/bin/awk -f
BEGIN {
    CONVERT[0] = 1
    CONVERT[1] = 2
    CONVERT[2] = 3
    CONVERT[3] = 4
    CONVERT[4] = 5
    CONVERT[5] = 6
}
{
    num = $1
    $1 = ""
    printf("%d%s\n", CONVERT[num], $0)
}
