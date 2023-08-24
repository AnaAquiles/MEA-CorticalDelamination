#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:18:57 2021

@author: aaquiles
"""
import sys, importlib, os
import McsPy.McsData
import McsPy.McsCMOS
import McsPy.McsCMOSMEA as McsCMOSMEA
from McsPy import ureg, Q_
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
p = Path('AP003_p30_60_200_basal.mcd')
p.rename(p.with_suffix('.h5'))

# path2TestData = r'misc/carr/aaquiles/Mea' # adjust this!
# path2TestDataFile1 = McsPy.McsData.RawData(os.path.join(path2TestData, "2014-07-09T10-17-35W8 Standard all 500 Hz.h5"))


channel_raw_data = McsPy.McsData.RawData("AP002_p30_120_200_cero.h5")


# channel_raw_data = McsPy.McsData.RawData('AP002_p30_120_200_cero.h5')
                                         

#%%
with open("log.txt") as infile:
    for line in infile:
        do_something_with(line)

