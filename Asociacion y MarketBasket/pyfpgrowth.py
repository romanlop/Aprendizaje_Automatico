#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 12:47:11 2019

@author: Ruman
"""

import pyfpgrowth

dataset = [['Pan', 'Leche'],
           ['Pan', 'Pañales', 'Cerveza', 'Huevos'],
           ['Leche', 'Pañales', 'Cerveza', 'Cola'],
           ['Leche', 'Pan', 'Pañales', 'Cerveza'],
           ['Pañales', 'Pan', 'Leche', 'Cola'],
           ['Pan', 'Leche', 'Pañales'],
           ['Pan', 'Cola']]

patterns = pyfpgrowth.find_frequent_patterns(dataset, 4)
patterns

rules = pyfpgrowth.generate_association_rules(patterns, 0.75)
rules
