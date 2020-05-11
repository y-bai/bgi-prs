#!/usr/bin/env sh

# =====================================================
# Description: main.sh
#
# =====================================================
#
# Created by YongBai on 2020/4/27 10:28 AM.

echo "python main.py">run_1.sh

qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=10g,p=1 run_1.sh