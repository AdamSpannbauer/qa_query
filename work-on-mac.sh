#!/bin/bash

PYVERSION=$(python --version)

if [ "$PYVERSION" != "Python 3.6.5" ]; then
    echo Your python version is not 3.6.5, only python 3.6.5 is confirmed to work
fi

echo "Correct version of Python, 3.6.5, detacted."

read -n 1 -p "Press any key to run 'pip install -r requirements.txt' and install the requirements" dummy
pip install -r requirements.txt 


read -n 1 -p "Press any key to run 'python -m deeppavlov install squad_bert' and install berts" dummy
python -m deeppavlov install squad_bert

echo Now you need to install nltk. Run the command:
echo
echo python nltk_download.py
echo
echo I had to run over 10 times due to timeouts. Once crashed you may need to use ctrl-c to exit the python
echo
read -n 1 -p "Press any once has completed successfully:" dummy

read -n 1 -p "Press any key to run the sanity check which will also download the library" dummy
python sanity-check.py

