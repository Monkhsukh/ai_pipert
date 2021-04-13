#!/bin/sh



echo "Building PIPERT ..."

# We do it this way so that we can abstract if from just git later on
if [ ! -d pipert/.git ]
then
    echo "Clonning!"
    git clone https://github.com/gerazo/pipert.git 
    cd pipert
else
    echo "Repo found!"
    cd pipert
    git pull https://github.com/gerazo/pipert.git
fi

# End

./build.sh

echo "Done."
