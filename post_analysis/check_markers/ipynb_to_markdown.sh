#!/bin/bash

cd notebooks/Mouse_Dentate_Gyrus

jupyter nbconvert --to markdown Emx1_Ctrl.ipynb
jupyter nbconvert --to markdown Nestin_Ctrl.ipynb
jupyter nbconvert --to markdown Emx1_Mut.ipynb
jupyter nbconvert --to markdown Nestin_Mut.ipynb

cd ../Mouse_Isocortex_Hippocampus

jupyter nbconvert --to markdown Emx1_Ctrl.ipynb
jupyter nbconvert --to markdown Nestin_Ctrl.ipynb
jupyter nbconvert --to markdown Emx1_Mut.ipynb
jupyter nbconvert --to markdown Nestin_Mut.ipynb