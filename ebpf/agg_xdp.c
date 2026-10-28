#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <linux/pkt_cls.h>
#include <arpa/inet.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include "agg_common.h"

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __type(key, __u32);
    __type(value, struct agg_map);
    __uint(max_entries, FRAGMENT_SIZE);
    __uint(pinning, LIBBPF_PIN_BY_NAME);
} aggregator_map SEC(".maps");

//struct {
//    __uint(type, BPF_MAP_TYPE_XSKMAP);
//    __uint(max_entries, 64);   // max #queues
//    __type(key, __u32);
//    __type(value, __u32);
//} xsks_map SEC(".maps");


SEC("xdp/aggregator")
int aggregator(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;
    struct ethhdr *eth = data;
    struct iphdr *ip = (struct iphdr *)(eth + 1);
    struct udphdr *udp = (struct udphdr *)(ip + 1);
    unsigned char hmac[6] = { HOST_MAC };
    unsigned char dmac[6] = { PARENT_MAC };

    if (udp + 1 > (struct udphdr *)data_end)
        return XDP_PASS;

    if (ip->protocol != IPPROTO_UDP)
        return XDP_PASS;
    
    if (udp->dest != htons(PORT))
        return XDP_PASS;

    struct agg_payload *payload = (struct agg_payload *)(udp + 1);
    if (payload + 1 > (struct agg_payload *)data_end) 
        return XDP_DROP;
    
    __u32 host_id = ntohl(payload->hid);
    __u32 frag_id = ntohl(payload->fid);
    __u32 iter = ntohl(payload->iter);
    __u32 bcast = htonl(payload->bcast);
    
    struct agg_map *map = bpf_map_lookup_elem(&aggregator_map, &frag_id);
    if (!map)
        return XDP_DROP;
    
    // forward to tc hook to broadcast
    if (bcast > 0 && ip->saddr == htonl(HOST_IP))
        return XDP_PASS;

    // send back for pkt lost
//    if (map->iter == iter + 1) {
//        payload->hid = htonl(HOST_ID);
//        for (__u32 i = 0; i < GRADIENT_SIZE; i++)
//            payload->grads[i] = htonl(map->grads[i]);
//
//        ip->daddr = ip->saddr;
//        ip->saddr = htonl(HOST_IP);
//        ip->ttl--;
//        compute_ipv4_csum(ip);
//        compute_udp_csum(udp, ip->saddr, ip->daddr);
//        __memcpy(eth->h_dest, eth->h_source, 6);
//        __memcpy(eth->h_source, hmac, 6);
//        return XDP_TX;
//    }
    
    if (map->iter != iter)
        return XDP_DROP;

    // aggregation
    if (map->childcnt <= CHILDREN_NUM) {
        map->lock = 1;
        
        // update aggregated node
        for (int worker_id = 0; worker_id < WORKER_NUM; worker_id++) {
            if (worker_id == host_id) {
                if (map->hcheck[worker_id]) {
		            map->lock = 0;
		            return XDP_DROP;
		        }
                map->hcheck[worker_id] = 1;
                break;
            }
        }
        
        // upadate local grads
        for (int i = 0; i < GRADIENT_SIZE; i++)
            map->lgrads[i] += payload->grads[i];
    	
        map->childcnt++;
        map->lock = 0;
    }

    // finish aggregation
    if (map->childcnt >= CHILDREN_NUM + 1) {
        payload->hid = htonl(HOST_ID);
        ip->ttl--;

        // parameter server to children
        if (HOST_ID == PARENT_ID) {
            map->lock = 1;
            map->childcnt = 0;
            map->iter++;
            
            for (int i = 0; i < WORKER_NUM; i++)
                map->hcheck[i] = 0;
            
            for (int i = 0; i < GRADIENT_SIZE; i++) {
                payload->grads[i] = map->lgrads[i];
                map->grads[i] = map->lgrads[i];
                map->lgrads[i] = 0;
            }
            
            payload->bcast = 0;
            map->lflag = 1;
	        map->lock = 0;
            
            return XDP_PASS; // forward to TC hook for multicast
        }
	
        // aggregator to children
        if (host_id == PARENT_ID) {
	        map->lock = 1;
            map->childcnt = 0;
            map->iter++;
            
            for (int i = 0; i < WORKER_NUM; i++)
                map->hcheck[i] = 0;
            
            for (int i = 0; i < GRADIENT_SIZE; i++) {
                map->grads[i] = payload->grads[i];
                map->lgrads[i] = 0;
            }
            
            payload->bcast = 0;
            map->lflag = 1;
	        map->lock = 0;
            
            return XDP_PASS;
        }

        // children to parent
        if (PARENT_NUM > 0) {
            for (int i = 0; i < GRADIENT_SIZE; i++)
                payload->grads[i] = map->lgrads[i];

            ip->saddr = htonl(HOST_IP);
            ip->daddr = htonl(PARENT_IP);
            
            compute_ipv4_csum(ip);
            compute_udp_csum(udp, ip->saddr, ip->daddr);
            
            __memcpy(eth->h_source, hmac, 6);
            __memcpy(eth->h_dest, dmac, 6);

//            __u32 index = ctx->rx_queue_index;
//            if (bpf_map_lookup_elem(&xsks_map, &index))
//                return bpf_redirect_map(&xsks_map, index, 0);

            return XDP_PASS;
        }
    }

    return XDP_DROP;
}

char _license[] SEC("license") = "GPL";
