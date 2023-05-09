# ！/usr/bin/python3
# -*- coding: utf-8 -*-
# @TIme 2023/5/9 20:46
# @Author: SAM
# @File: overfit_and_underfit.py
# @Email: SAM-Turentu@outlook.com
# @Desc: 过拟合和欠拟合


import pathlib
import shutil
import tempfile

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from IPython import display

# pip install ipython
# pip install git+https://github.com/tensorflow/docs
def main():
    logdir = pathlib.Path(tempfile.mkdtemp()) / "tensorboard_logs"
    shutil.rmtree(logdir, ignore_errors=True)


if __name__ == '__main__':
    main()
