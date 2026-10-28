#!/bin/bash

fdir=`dirname $0`
ifnet=ens3

#echo sudo pkill -9 agg_xdp_usr
#sudo pkill -9 agg_xdp_usr

echo sudo pkill -f "python3 worker.py"
sudo pkill -f "python3 worker.py"

echo sudo tc qdisc del dev $ifnet clsact
sudo tc qdisc del dev $ifnet clsact
#echo sudo tc filter del dev $ifnet ingress
#sudo tc filter del dev $ifnet ingress

#echo sudo ip link set dev $ifnet xdp off
#sudo ip link set dev $ifnet xdp off
echo sudo xdp-loader unload --all $ifnet
sudo xdp-loader unload --all $ifnet

echo sudo rm /sys/fs/bpf/*_map
sudo rm /sys/fs/bpf/*_map

echo sudo rm $fdir/../ebpf/*.o $fdir/../ebpf/*.so
sudo rm $fdir/../ebpf/*.o $fdir/../ebpf/*.so
