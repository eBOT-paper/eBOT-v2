#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <getopt.h>
#include <unistd.h>
#include <assert.h>
#include <sys/resource.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <xdp/libxdp.h>
#include <xdp/xsk.h>

#include <arpa/inet.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include "agg_usr.h"
#include "agg_common.h"


const char *agg_map_name = "aggregator_map";

int pin_map_in_bpf_object(struct bpf_object *bpf_obj, const char *subdir, const char *map_name) {
    char map_filename[PATH_MAX];
    char pin_dir[PATH_MAX];
    int err, len;

    len = snprintf(pin_dir, PATH_MAX, "%s/%s", pin_basedir, subdir);
    if (len < 0) {
        fprintf(stderr, "ERR: creating pin dirname\n");
        return EXIT_FAIL_OPTION;
    }

    len = snprintf(map_filename, PATH_MAX, "%s/%s/%s", pin_basedir, subdir, map_name);
    if (len < 0) {
        fprintf(stderr, "ERR: creating map_name\n");
        return EXIT_FAIL_OPTION;
    }


    if (access(map_filename, F_OK) != -1) {
        printf(" - Unpinning (remove) prev map %s in %s/\n", map_name, pin_dir);

        err = bpf_object__unpin_maps(bpf_obj, pin_dir);
        if (err) {
            fprintf(stderr, "ERR: Unpinning map %s in %s\n", map_name, pin_dir);
            return EXIT_FAIL_BPF;
        }
    }

    printf(" - Pinning map %s in %s/\n", map_name, pin_dir);
    err = bpf_object__pin_maps(bpf_obj, pin_dir);
    if (err) 
        return EXIT_FAIL_BPF;
    
    return 0;
}

int main(int argc, char **argv) {
	struct xdp_program *prog;
    struct bpf_object *bpf_obj;
	char errmsg[1024];
	int err;

	struct config cfg = {
		//.attach_mode = XDP_MODE_UNSPEC,
		.attach_mode = XDP_MODE_SKB,
		.xdp_flags = 0,
        .ifname    = "ens3",
        .filename  = "eBOT/agg_xdp.o",
        .progname  = "aggregator"
	};

	cfg.ifindex = if_nametoindex(cfg.ifname);

	DECLARE_LIBBPF_OPTS(bpf_object_open_opts, bpf_opts);
	DECLARE_LIBXDP_OPTS(xdp_program_opts, xdp_opts,
                            .open_filename = cfg.filename,
                            .prog_name = cfg.progname,
                            .opts = &bpf_opts);

	prog = xdp_program__create(&xdp_opts);
	err = libxdp_get_error(prog);
	if (err) {
		libxdp_strerror(err, errmsg, sizeof(errmsg));
		fprintf(stderr, "Couldn't get XDP program %s: %s\n",
			cfg.progname, errmsg);
		return err;
	}

	err = xdp_program__attach(prog, cfg.ifindex, cfg.attach_mode, 0);
	if (err) {
		libxdp_strerror(err, errmsg, sizeof(errmsg));
		fprintf(stderr, "Couldn't attach XDP program on iface '%s' : %s (%d)\n",
			cfg.ifname, errmsg, err);
		return err;
	}

	printf("Success: Loading "
	       "XDP prog on device:%s(ifindex:%d)\n", cfg.ifname, cfg.ifindex);
	
//    err = pin_map_in_bpf_object(xdp_program__bpf_obj(prog), cfg.ifname, agg_map_name);
//    if (err) {
//        fprintf(stderr, "ERR: pinning map %s\n", agg_map_name);
//        return err;
//    }


    return EXIT_OK;
}
