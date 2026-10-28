CLANG ?= clang

all: build

build: agg_tc.o agg_xdp_usr agg_map_lib.so #af_xdp_usr

agg_tc.o: agg_tc.c
	$(CLANG) -g -O2 -target bpf -c agg_tc.c

agg_xdp.o: agg_xdp.c
	$(CLANG) -g -O2 -target bpf -c agg_xdp.c

agg_xdp_usr: agg_xdp.o agg_xdp_usr.c
	$(CLANG) -g -O2 -o agg_xdp_usr agg_xdp_usr.c -lpthread -lbpf -lxdp

agg_map_lib.so: agg_map_lib.c
	$(CLANG) -shared -o agg_map_lib.so agg_map_lib.c -lbpf -lxdp

#af_xdp_usr: af_xdp_usr.c agg_xdp.o
#	$(CLANG) -g -O2 -o af_xdp_usr af_xdp_usr.c -lbpf -lxdp
