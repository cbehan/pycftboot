#!/usr/bin/env python2
"""
This concatenates all of the PyCFTBoot files. The idea is to need only one command
to send it to a cluster.
"""
from __future__ import print_function
import shutil
import sys

i = 0
files = ["common.py", "compat_juliboots.py", "blocks1.py", "blocks2.py"]
main_file = open("bootstrap.py", 'r')

for line in main_file:
    if line[:4] != "exec":
        print(line, end = "")
    else:
        print("")
        f = open(files[i], 'r')
        shutil.copyfileobj(f, sys.stdout)
        f.close()
        i += 1
main_file.close()
