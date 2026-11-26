# POPSTAR (not included)

This directory is where `tools/get_popstar.sh` puts the original POPSTAR program of
M. G. C. Resende and R. F. Werneck (*A hybrid heuristic for the p-median problem*,
Journal of Heuristics 10:59–88, 2004).

POPSTAR is the authors' software and is **not redistributed** with this repository.
Get it from the authors' page, <https://mauricio.resende.info/popstar/downloads.html>:

```
tools/get_popstar.sh                    # downloads popstar.tar.gz and compiles it
tools/get_popstar.sh /path/popstar.tar.gz   # or use a copy you already downloaded
```

The script compiles the source unchanged (`g++ -O3 -std=gnu++98`) and leaves the executable in
`third_party/popstar/popstar`, which is where `mkbpx/popstar_orig.py` looks for it (or set the
environment variable `POPSTAR_BIN`).

The paper used version `20041104` of the program (the value it prints as `version`);
sha256 of the distribution: `37f0627c35481f5b76ed50756289f472579a918e1d2bb12803f06a0e2ec2f43d`.

The pre-compiled Linux executable offered on the same page is a 32-bit binary built in 2002; on a
current 64-bit system (glibc 2.39) it stops with a segmentation fault inside the dynamic loader
before the program starts, so it cannot be used. The source of the same distribution is compiled instead.
