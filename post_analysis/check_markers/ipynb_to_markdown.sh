#!/bin/bash

cd notebooks

jupyter nbconvert --to markdown Emx1_Ctrl.ipynb
jupyter nbconvert --to markdown Nestin_Ctrl.ipynb
jupyter nbconvert --to markdown Emx1_Mut.ipynb
jupyter nbconvert --to markdown Nestin_Mut.ipynb
