#!/bin/bash

echo "==== init submodule 'sdk' ===="
git submodule init
git submodule update

echo "==== install require packages ===="
sudo apt update
sudo apt install -y python3.11 python3.11-dev python3-virtualenv
sudo dpkg -i sdk/packages/*.deb

echo "==== init virtual environment ===="
virtualenv venv --python=python3.11
source venv/bin/activate
pip install sdk/packages/hailort-*.whl
pip install -r sdk/requirements.txt

echo "==== done ===="
echo "now you can try \"source venv/bin/activate\" to enable virtual environment"
echo "after then, execute the script at \"scripts/start-hailo-object-detection.sh\""
echo "see README.md for more information"
echo ""

echo "==== have any question ==="
echo "Please contact Sales or Account executive for help."
echo ""

