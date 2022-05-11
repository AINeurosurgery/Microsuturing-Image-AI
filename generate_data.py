#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 11:24:46 2021

@author: rohan
"""

import os
import cv2
import shutil
path = 'Micro_suturing_Images/'
out_path = 'DATA/training/images/'
if not os.path.isdir(out_path):
    os.makedirs(out_path)
p = os.listdir(path)
for i in p:
    if os.path.isdir(path+i):
        new_path = os.path.join(path,i)
        #print(new_path)
        for j in new_path:
            p1 = os.listdir(new_path)
            for l in p1:
                if l[-1]=='g' and not l[0]=='.':
                    #I = cv2.imread(os.path.join(new_path,l))
                    new_name = i+'_'+l
                    shutil.copy(os.path.join(new_path,l),os.path.join(out_path,new_name))
                    
                    
                    