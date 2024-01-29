# Read a sub file which has lines of format {frame1}{frame2}xxxxx
# some lines don't begin with {frame1}{frame2} and are just xxxxx
# we want to move those lines to the previous line, with a literal \n in between

import sys
import re

if __name__ == '__main__':
    # get name of file
    if len(sys.argv) < 3:
        print("Usage: python subfile_newlines.py <infile> <outfile>")
        sys.exit(1)
    infile = sys.argv[1]
    # open file
    with open(infile, 'r') as f:
        lines = f.readlines()
    # iterate through lines
    new_lines = []
    for line in lines:
        # if line starts with {frame1}{frame2}, add it to new_lines
        if re.match(r'^\{.*\}\{.*\}', line):
            new_lines.append(line)
        # if line doesn't start with {frame1}{frame2}, add it to the previous line
        else:
            new_lines[-1] = new_lines[-1].rstrip() + r'\n' + line

    # write new lines to file
    outfile = sys.argv[2]
    with open(outfile, 'w') as f:
        f.writelines(new_lines)