#!/bin/bash

cd notebooks

jupyter nbconvert --to markdown Mouse_Dentate_Gyrus_Emx1_Ctrl.ipynb
jupyter nbconvert --to markdown Mouse_Dentate_Gyrus_Emx1_Mut.ipynb
jupyter nbconvert --to markdown Mouse_Dentate_Gyrus_Nestin_Ctrl.ipynb
jupyter nbconvert --to markdown Mouse_Dentate_Gyrus_Nestin_Mut.ipynb
jupyter nbconvert --to markdown Mouse_Isocortex_Hippocampus_Emx1_Ctrl.ipynb
jupyter nbconvert --to markdown Mouse_Isocortex_Hippocampus_Emx1_Mut.ipynb
jupyter nbconvert --to markdown Mouse_Isocortex_Hippocampus_Nestin_Ctrl.ipynb
jupyter nbconvert --to markdown Mouse_Isocortex_Hippocampus_Nestin_Mut.ipynb