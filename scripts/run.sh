#!/bin/bash

if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root. Use: sudo $0"
    exit 1
fi

echo "Running with sudo privileges!"

RED='\e[31m'
GREEN='\e[32m'
RESET='\e[0m'
filedir=`dirname $0`
ifnet=ens3

echo -e "\n${GREEN}./scripts/clean.sh${RESET} $ifnet"
$filedir/clean.sh $ifnet

echo -e "\n${GREEN}make -C $filedir/../ebpf${RESET}"
make -C $filedir/../ebpf

echo -e "\n${GREEN}./scripts/xdprun.sh${RESET} $ifnet"
$filedir/xdprun.sh $ifnet

echo -e "\n${GREEN}./scripts/tcrun.sh${RESET} $ifnet"
$filedir/tcrun.sh $ifnet
