FULL2HALF = dict((i + 0xFEE0, i) for i in range(0x21, 0x7F))
FULL2HALF[0x3000] = 0x20
HALF2FULL = dict((i, i + 0xFEE0) for i in range(0x21, 0x7F))
HALF2FULL[0x20] = 0x3000

def halfen(s):
    return str(s).translate(FULL2HALF)

def fullen(s):
    return str(s).translate(HALF2FULL)

def is_full_width(s):
    return 0xFF01 <= ord(s) <= 0xFF5E


def is_half_width(s):
    return 0x21 <= ord(s) <= 0x7E