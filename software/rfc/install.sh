#! /usr/bin/env bash
cd RFsubs
/opt/conda/bin/gfortran -c -O3 *.f*
cd ..
/opt/conda/bin/f2py -m rfc -c RF.F90 RFsubs/*.f*
