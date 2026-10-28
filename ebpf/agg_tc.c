#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <linux/pkt_cls.h>
#include <arpa/inet.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include "agg_common.h"


SEC("tc/ingress/broadcast")
int broadcast(struct __sk_buff *skb) {
    void *data = (void *)(long)skb->data;
    void *data_end = (void *)(long)skb->data_end;
    struct ethhdr *eth = data;
    struct iphdr *ip = (struct iphdr *)(eth + 1);
    struct udphdr *udp = (struct udphdr *)(ip + 1);
    unsigned char maclist[CHILDREN_NUM * 6] = { CHILDREN_MAC };
    unsigned char smac[6] = { HOST_MAC };
    __u32 children[CHILDREN_NUM] = { CHILDREN_IP };
    __u32 if_index = 2;

    if (udp + 1 > (struct udphdr *)data_end)
         return TC_ACT_OK;

    if (ip->protocol != IPPROTO_UDP)
        return TC_ACT_OK;
    
    if (udp->dest != htons(PORT))
        return TC_ACT_OK;
    
    struct agg_payload *payload = (struct agg_payload *)(udp + 1);
    if (payload + 1 > (struct agg_payload *)data_end) 
        return TC_ACT_SHOT;

    // forward to parent except for parameter server
    if (ip->daddr == htonl(PARENT_IP) && HOST_IP != PARENT_IP) {
        return bpf_redirect(if_index, 0);
    }
    
    __u32 bcast = ntohl(payload->bcast);
    if (bcast >= CHILDREN_NUM)
        return TC_ACT_SHOT;
        
    payload->bcast = htonl(bcast + 1);
    ip->saddr = htonl(HOST_IP);

    for (int child_id = 0; child_id < CHILDREN_NUM; child_id++) {
        if (child_id == bcast) {
            ip->daddr = htonl(children[child_id]);
        }
    }

    compute_ipv4_csum(ip);
    compute_udp_csum(udp, ip->saddr, ip->daddr);
    __memcpy(eth->h_source, smac, 6);
    __memcpy(eth->h_dest, maclist + bcast * 6, 6);

    // Due to ebpf verifier
    // we cannot change data in the packet more than two times
    // so we use a trick 
    bpf_clone_redirect(skb, if_index, BPF_F_INGRESS); // copy to ingress
    return bpf_redirect(if_index, 0);
    //bpf_clone_redirect(skb, if_index, 0); // copy to egress
    //return TC_ACT_SHOT; // drop the original packet
}


SEC("tc/egress/local")
int local(struct __sk_buff *skb) {
    void *data = (void *)(long)skb->data;
    void *data_end = (void *)(long)skb->data_end;
    struct ethhdr *eth = data;
    struct iphdr *ip = (struct iphdr *)(eth + 1);
    struct udphdr *udp = (struct udphdr *)(ip + 1);
    __u32 if_index = 2;

    if (udp + 1 > (struct udphdr *)data_end)
         return TC_ACT_OK;
    
    if (ip->protocol != IPPROTO_UDP)
        return TC_ACT_OK;

    if (udp->dest != htons(PORT))
        return TC_ACT_OK;
    
    if (ip->daddr == htonl(DUMMY_IP)) {
        ip->daddr = htonl(HOST_IP);
        //bpf_clone_redirect(skb, if_index, BPF_F_INGRESS);
        //return TC_ACT_SHOT;
        return bpf_redirect(if_index, BPF_F_INGRESS);
    } 

    return TC_ACT_OK;
}

char LICENSE[] SEC("license") = "Dual BSD/GPL";
