# ！/usr/bin/python3
# -*- coding: utf-8 -*-
# @TIme 2023/5/5 19:05
# @Author: SAM
# @File: __init__.py.py
# @Email: SAM-Turentu@outlook.com
# @Desc:


import os

if 'BaseLearnTF' in os.path.dirname(os.getcwd()):
    ProjectPath = os.path.abspath(os.path.join(os.getcwd(), "../.."))
else:
    ProjectPath = os.path.dirname(os.getcwd())
