#!/bin/bash

filedir=`dirname $0`
filename=ebpf/agg_tc
ifnet=$1

# install tc program
echo sudo tc qdisc add dev $ifnet clsact
sudo tc qdisc add dev $ifnet clsact

echo sudo tc filter add dev $ifnet ingress bpf da obj $filedir/../$filename.o sec tc/ingress/broadcast
sudo tc filter add dev $ifnet ingress bpf da obj $filedir/../$filename.o sec tc/ingress/broadcast

echo sudo tc filter add dev $ifnet egress bpf da obj $filedir/../$filename.o sec tc/egress/local
sudo tc filter add dev $ifnet egress bpf da obj $filedir/../$filename.o sec tc/egress/local

# show program
#tc filter show dev $ifnet ingress
