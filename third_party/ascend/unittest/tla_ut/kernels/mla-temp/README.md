# MLA verification and tuning archive

`original` preserves the old experiment source, all intermediate schedules and
the 112-token pipeline variant. `evidence` contains the accepted raw timing
samples, traces and logs. `upstream` records the published compiler branches.
The setup, extraction and comparison scripts in this directory are development
tools. The two final operators can run from their own directories.

The packaged `ir` directory contains 950PR and 950DT compiler replay inputs,
full pass dumps, final IR and ELF files. The packaged `toolchain` directory
contains the matched Triton wheel and a separate NPU-IR installation prefix.

The previously measured hardware timings used the original BF16 L0/O2
toolchain. The new toolchain was validated by compiling the two final sources
for both targets and by running targeted frontend and NPU-IR tests.
