#!/bin/bash

for d in */ ; do
    echo "$d"
    cd $d
    python -m unittest discover -v
    cd ../
done
