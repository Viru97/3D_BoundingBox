#!/usr/bin/env bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
python scripts/"$@"
