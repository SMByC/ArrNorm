#!/usr/bin/env bash

if [ ! -d ${ARRNORM_DIR} ]; then
    mkdir -p $ARRNORM_DIR
fi

cd $ARRNORM_DIR

# check if the project arrNorm exist with VCS
if [ ! -d ${ARRNORM_DIR}/.hg ]; then
    cd ..
    rm -rf ${ARRNORM_DIR}
    hg clone https://bitbucket.org/SMBYC/arrnorm arrNorm
    cd $ARRNORM_DIR
fi

# synchronize changes with repository, update and clean
hg pull
hg update -C
hg status -un|xargs rm 2> /dev/null

# print status
echo -e "\nThe last commit:\n"
hg tip
echo -e "Update finished\n"
