#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <bpf/bpf.h>
#include <xdp/libxdp.h>
#include "agg_common.h"


static char *aggmap_path = "/sys/fs/bpf/aggregator_map";

int get_aggmap_fd() {
    int agg_fd = bpf_obj_get(aggmap_path);
    if (agg_fd < 0) {
        printf("Failed to get agg_map fd, fd = %d\n", agg_fd);
        return agg_fd;
    }

    return agg_fd;
}

struct agg_map get_aggmap(int agg_fd, int frag_id) {
    struct agg_map map;
    bpf_map_lookup_elem(agg_fd, &frag_id, &map);
    return map;
}

int* busy_polling(int agg_fd, int prev_iter) {
    static int all_grads[FRAGMENT_SIZE * GRADIENT_SIZE];
    struct agg_map map;
    bool updated[FRAGMENT_SIZE] = { false };
    int count = 0;

    memset(all_grads, 0, sizeof(all_grads));

    while (count < FRAGMENT_SIZE) {
	    for (int frag_id = 0; frag_id < FRAGMENT_SIZE; frag_id++) {
	        if (updated[frag_id]) continue;

            bpf_map_lookup_elem(agg_fd, &frag_id, &map);

            if (map.lock == 1) continue;

            if (map.iter == prev_iter + 1 && map.lflag == 1) {
                for (size_t i = 0; i < GRADIENT_SIZE; i++) {
                    int32_t val = map.grads[i];
                    all_grads[frag_id * GRADIENT_SIZE + i] = map.grads[i];
                }

                map.lflag = 0;
                bpf_map_update_elem(agg_fd, &frag_id, &map, 0);
                
                updated[frag_id] = true;
                count++;
            }
        }
        usleep(5);
    }

    return all_grads;
}


static int sockfd = -1;
static struct sockaddr_in dest_addr;

int init_socket(const char *ifname) {
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("socket");
        return -1;
    }

    if (setsockopt(sockfd, SOL_SOCKET, SO_BINDTODEVICE, ifname, strlen(ifname) + 1) < 0) {
        perror("SO_BINDTODEVICE");
        close(sockfd);
        sockfd = -1;
        return -1;
    }

    memset(&dest_addr, 0, sizeof(dest_addr));
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(PORT);
    dest_addr.sin_addr.s_addr = htonl(DUMMY_IP);

    return 0;
}

int send_all_fragments(int step, const int32_t *grads) {
    if (sockfd < 0) {
        fprintf(stderr, "Socket not initialized\n");
        return -1;
    }
    
    static uint8_t payload[sizeof(uint32_t) * 4 + GRADIENT_SIZE * sizeof(int32_t)];

    for (int frag_id = 0; frag_id < FRAGMENT_SIZE; frag_id++) {
        uint32_t header[4];
        header[0] = htonl(HOST_ID);
        header[1] = htonl(frag_id);
        header[2] = htonl(0); //bcast_id
        header[3] = htonl(step);

        size_t grad_offset = frag_id * GRADIENT_SIZE;
        size_t grad_bytes = GRADIENT_SIZE * sizeof(int32_t);
        size_t payload_size = sizeof(header) + grad_bytes;

        memcpy(payload, header, sizeof(header));

        memcpy(payload + sizeof(header), grads + grad_offset, grad_bytes);

        ssize_t sent = sendto(sockfd, payload, payload_size, 0,
                (struct sockaddr *)&dest_addr, sizeof(dest_addr));
        if (sent < 0) {
            perror("sendto");
            return -1;
        }
    }

    return 0;
}

void close_socket() {
    if (sockfd >= 0) {
        close(sockfd);
        sockfd = -1;
    }
}

void free_buffer(void* ptr) {
    free(ptr);
}
