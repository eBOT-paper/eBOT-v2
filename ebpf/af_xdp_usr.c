#include <signal.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>
#include <sys/resource.h>
#include <unistd.h>
#include <time.h>
#include <locale.h>
#include <pthread.h>

#include <net/if.h>
#include <arpa/inet.h>
#include <linux/if_xdp.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>

#include <xdp/libxdp.h>
#include <xdp/xsk.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

//#define NUM_FRAMES 4096
//#define FRAME_SIZE XSK_UMEM__DEFAULT_FRAME_SIZE
//#define RX_BATCH_SIZE 64
#define NUM_FRAMES 16384
#define FRAME_SIZE 32768
#define RX_BATCH_SIZE 64
#define INVALID_UMEM_FRAME UINT64_MAX
#define NANOSEC_PER_SEC 1000000000 /* 10^9 */

static struct xdp_program *prog;
static bool global_exit;
int xsk_map_fd;
struct config cfg;

struct config {
  enum xdp_attach_mode attach_mode;
  __u32 xdp_flags;
  __u16 xsk_bind_flags;
  char *ifname;
  int ifindex;
  int xsk_if_queue;
  bool verbose;
};

struct xsk_umem_info {
  struct xsk_ring_cons cq;
  struct xsk_ring_prod fq;
  struct xsk_umem *umem;
  void *buffer;
};

struct stats_record {
  uint64_t timestamp;
  uint64_t rx_packets;
  uint64_t rx_bytes;
  uint64_t tx_packets;
  uint64_t tx_bytes;
};

struct xsk_socket_info {
  struct xsk_ring_cons rx;
  struct xsk_ring_prod tx;
  struct xsk_umem_info *umem;
  struct xsk_socket *xsk;

  uint64_t umem_frame_addr[NUM_FRAMES];
  uint32_t umem_frame_free;
  uint32_t outstanding_tx;

  struct stats_record stats;
  struct stats_record prev_stats;
};

static struct xsk_umem_info *configure_xsk_umem(void *buffer, uint64_t size) {
  struct xsk_umem_info *umem;
  int ret;

  umem = calloc(1, sizeof(*umem));
  if (!umem) return NULL;

  ret = xsk_umem__create(&umem->umem, buffer, size, &umem->fq, &umem->cq, NULL);
  if (ret) {
    errno = -ret;
    return NULL;
  }
  umem->buffer = buffer;
  return umem;
}

static uint64_t xsk_alloc_umem_frame(struct xsk_socket_info *xsk) {
  uint64_t frame;
  if (xsk->umem_frame_free == 0)
    return INVALID_UMEM_FRAME;

  frame = xsk->umem_frame_addr[--xsk->umem_frame_free];
  xsk->umem_frame_addr[xsk->umem_frame_free] = INVALID_UMEM_FRAME;
  return frame;
}

static void xsk_free_umem_frame(struct xsk_socket_info *xsk, uint64_t frame) {
  assert(xsk->umem_frame_free < NUM_FRAMES);
  xsk->umem_frame_addr[xsk->umem_frame_free++] = frame;
}

static uint64_t xsk_umem_free_frames(struct xsk_socket_info *xsk) {
  return xsk->umem_frame_free;
}

static struct xsk_socket_info *xsk_configure_socket(struct config *cfg, struct xsk_umem_info *umem) {
  struct xsk_socket_config xsk_cfg;
  struct xsk_socket_info *xsk_info;
  int ret, i;
  uint32_t prog_id, idx;

  xsk_info = calloc(1, sizeof(*xsk_info));
  if (!xsk_info)
    return NULL;

  xsk_info->umem = umem;
  xsk_cfg.rx_size = XSK_RING_CONS__DEFAULT_NUM_DESCS;
  xsk_cfg.tx_size = XSK_RING_PROD__DEFAULT_NUM_DESCS;
  xsk_cfg.xdp_flags = cfg->xdp_flags;
  xsk_cfg.bind_flags = cfg->xsk_bind_flags;
  xsk_cfg.libbpf_flags = XSK_LIBBPF_FLAGS__INHIBIT_PROG_LOAD;
  ret = xsk_socket__create(&xsk_info->xsk, cfg->ifname, cfg->xsk_if_queue, umem->umem, &xsk_info->rx, &xsk_info->tx, &xsk_cfg);
  if (ret)
    goto error_exit;

  ret = xsk_socket__update_xskmap(xsk_info->xsk, xsk_map_fd);
  if (ret)
    goto error_exit;

  for (i = 0; i < NUM_FRAMES; i++)
    xsk_info->umem_frame_addr[i] = i * FRAME_SIZE;

  xsk_info->umem_frame_free = NUM_FRAMES;

  ret = xsk_ring_prod__reserve(&xsk_info->umem->fq, XSK_RING_PROD__DEFAULT_NUM_DESCS, &idx);
  if (ret != XSK_RING_PROD__DEFAULT_NUM_DESCS)
    goto error_exit;

  for (i = 0; i < XSK_RING_PROD__DEFAULT_NUM_DESCS; i++)
    *xsk_ring_prod__fill_addr(&xsk_info->umem->fq, idx++) = xsk_alloc_umem_frame(xsk_info);

  xsk_ring_prod__submit(&xsk_info->umem->fq, XSK_RING_PROD__DEFAULT_NUM_DESCS);

  return xsk_info;

error_exit:
  errno = -ret;
  return NULL;
}

static uint64_t gettime(void) {
  struct timespec t;
  int res;

  res =  clock_gettime(CLOCK_MONOTONIC, &t);
  if (res < 0) {
    fprintf(stderr, "ERROROR: Error with gettimeofday! (%i)\n", res);
    exit(EXIT_FAILURE);
  }

  return (uint64_t) t.tv_sec * NANOSEC_PER_SEC + t.tv_nsec;
}

static double calc_period(struct stats_record *r, struct stats_record *p) {
  double period_ = 0;
  __u64 period = 0;

  period = r->timestamp - p->timestamp;
  if (period > 0)
    period_ = ((double) period / NANOSEC_PER_SEC);

  return period_;
}

static void stats_print(struct stats_record *stats_rec, struct stats_record *stats_prev) {
  uint64_t packets, bytes;
  double period;
  double pps;
  double bps;

  char *fmt = "%-12s %'11lld pkts (%'10.0f pps)"
              " %'11lld Kbytes (%'6.0f Mbits/s)"
              " period:%f\n";
  period = calc_period(stats_rec, stats_prev);
  if (period == 0)
    period = 1;

  packets = stats_rec->rx_packets - stats_prev->rx_packets;
  pps     = packets / period;

  bytes   = stats_rec->rx_bytes   - stats_prev->rx_bytes;
  bps     = (bytes * 8) / period / 1000000;

  printf(fmt, "AF_XDP RX:", stats_rec->rx_packets, pps,
  stats_rec->rx_bytes / 1000 , bps, period);

  packets = stats_rec->tx_packets - stats_prev->tx_packets;
  pps     = packets / period;

  bytes   = stats_rec->tx_bytes   - stats_prev->tx_bytes;
  bps     = (bytes * 8) / period / 1000000;

  printf(fmt, "       TX:", stats_rec->tx_packets, pps, stats_rec->tx_bytes / 1000 , bps, period);

  printf("\n");
}

static void *stats_poll(void *arg) {
  unsigned int interval = 2;
  struct xsk_socket_info *xsk = arg;
  static struct stats_record prev_stats = { 0 };

  prev_stats.timestamp = gettime();

  setlocale(LC_NUMERIC, "en_US");

  while (!global_exit) {
    sleep(interval);
    xsk->stats.timestamp = gettime();
    stats_print(&xsk->stats, &prev_stats);
    prev_stats = xsk->stats;
  }

  return NULL;
}

static inline __sum16 csum16_add(__sum16 csum, __be16 addend) {
  uint16_t res = (uint16_t)csum;
  res += (__u16)addend;
  return (__sum16)(res + (res < (__u16)addend));
}

static inline __sum16 csum16_sub(__sum16 csum, __be16 addend) {
  return csum16_add(csum, ~addend);
}

static inline void csum_replace2(__sum16 *sum, __be16 old, __be16 new) {
  *sum = ~csum16_add(csum16_sub(~(*sum), old), new);
}

static bool process_packet(struct xsk_socket_info *xsk, uint64_t addr, uint32_t len) {
  uint8_t *pkt = xsk_umem__get_data(xsk->umem->buffer, addr);
 
  int ret;
  uint32_t tx_idx = 0;
  uint8_t tmp_mac[ETH_ALEN];
  uint16_t tmp_port;
  struct in_addr tmp_ip;
  struct ethhdr *eth = (struct ethhdr *)pkt;
  struct iphdr *ip = (struct iphdr *)(eth + 1);
  struct udphdr *udp = (struct udphdr *)(ip + 1);

  //char *body = (char *)(udp + 1);
  //char *new_body = "Server OK\n";

  //printf("Source: %d, Dest: %d ,Message: %s\n", ntohs(udp->source), ntohs(udp->dest), body);

  memcpy(tmp_mac, eth->h_dest, ETH_ALEN);
  memcpy(eth->h_dest, eth->h_source, ETH_ALEN);
  memcpy(eth->h_source, tmp_mac, ETH_ALEN);
  
  memcpy(&tmp_ip, &ip->daddr, sizeof(tmp_ip));
  memcpy(&ip->daddr, &ip->saddr, sizeof(tmp_ip));
  memcpy(&ip->saddr, &tmp_ip, sizeof(tmp_ip));

  memcpy(&tmp_port, &udp->dest, sizeof(tmp_port));
  memcpy(&udp->dest, &udp->source, sizeof(tmp_port));
  memcpy(&udp->source, &tmp_port, sizeof(tmp_port));

  //csum_replace2(&udp->check, htonl(*(__be16 *)body), htonl(*(__be16 *)new_body));
  //udp->check = 0;
  //strncpy(body, new_body, strlen(new_body));
  //printf("Server message: %s\n", body);

  ret = xsk_ring_prod__reserve(&xsk->tx, 1, &tx_idx);
  if (ret != 1)
    return false;
  
  xsk_ring_prod__tx_desc(&xsk->tx, tx_idx)->addr = addr;
  xsk_ring_prod__tx_desc(&xsk->tx, tx_idx)->len = len;
  xsk_ring_prod__submit(&xsk->tx, 1);
  xsk->outstanding_tx++;

  xsk->stats.tx_bytes += len;
  xsk->stats.tx_packets++;

  return true;
}

static void complete_tx(struct xsk_socket_info *xsk) {
  unsigned int completed;
  uint32_t idx_cq;

  if (!xsk->outstanding_tx)
    return;

  sendto(xsk_socket__fd(xsk->xsk), NULL, 0, MSG_DONTWAIT, NULL, 0);

  completed = xsk_ring_cons__peek(&xsk->umem->cq, XSK_RING_CONS__DEFAULT_NUM_DESCS, &idx_cq);

  if (completed > 0) {
    for (int i = 0; i < completed; i++)
      xsk_free_umem_frame(xsk, *xsk_ring_cons__comp_addr(&xsk->umem->cq, idx_cq++));

    xsk_ring_cons__release(&xsk->umem->cq, completed);
    xsk->outstanding_tx -= completed < xsk->outstanding_tx ? completed : xsk->outstanding_tx;
  }
}

static void recvxdp(struct xsk_socket_info *xsk) {
  unsigned int rcvd, stock_frames, i;
  uint32_t idx_rx = 0, idx_fq = 0;
  int ret;

  rcvd = xsk_ring_cons__peek(&xsk->rx, RX_BATCH_SIZE, &idx_rx);
  if (!rcvd)
    return;

  stock_frames = xsk_prod_nb_free(&xsk->umem->fq, xsk_umem_free_frames(xsk));
  if (stock_frames > 0) {
    ret = xsk_ring_prod__reserve(&xsk->umem->fq, stock_frames, &idx_fq);
    while (ret != stock_frames)
      ret = xsk_ring_prod__reserve(&xsk->umem->fq, rcvd, &idx_fq);

    for (i = 0; i < stock_frames; i++)
      *xsk_ring_prod__fill_addr(&xsk->umem->fq, idx_fq++) = xsk_alloc_umem_frame(xsk);

    xsk_ring_prod__submit(&xsk->umem->fq, stock_frames);
  }

  for (i = 0; i < rcvd; i++) {
    uint64_t addr = xsk_ring_cons__rx_desc(&xsk->rx, idx_rx)->addr;
    uint32_t len = xsk_ring_cons__rx_desc(&xsk->rx, idx_rx++)->len;

    if (!process_packet(xsk, addr, len + 1000))
      xsk_free_umem_frame(xsk, addr);

    xsk->stats.rx_bytes += len;
  }

  xsk_ring_cons__release(&xsk->rx, rcvd);
  xsk->stats.rx_packets += rcvd;

  complete_tx(xsk);
}

int do_unload(struct config *cfg) {
  struct xdp_multiprog *mp = NULL;
  int err = EXIT_FAILURE;

  mp = xdp_multiprog__get_from_ifindex(cfg->ifindex);
  if (libxdp_get_error(mp)) {
    fprintf(stderr, "Unable to get xdp_dispatcher program: %s\n", strerror(errno));
    goto out;
  } else if (!mp) {
    fprintf(stderr, "No XDP program loaded on %s\n", cfg->ifname);
    mp = NULL;
    goto out;
  }

  err = xdp_multiprog__detach(mp);
  if (err) {
    fprintf(stderr, "Unable to detach xdp program: %s\n", strerror(-err));
    goto out;
  }

out:
  xdp_multiprog__close(mp);
  return err ? EXIT_FAILURE : EXIT_SUCCESS;
}

static void exit_application(int sig) {
  int err;
  err = do_unload(&cfg);
  global_exit = true;
}

int main() {
  int ret, err;
  char errmsg[1024];
  struct bpf_object *bpf_obj;
  struct bpf_map *map;
  struct rlimit rlim = {RLIM_INFINITY, RLIM_INFINITY};
  struct xsk_umem_info *umem;
  struct xsk_socket_info *xsk_socket;
  void *packet_buffer;
  uint64_t packet_buffer_size;
  pthread_t stats_poll_thread;

  cfg.ifname = "ens3";
  cfg.ifindex = if_nametoindex(cfg.ifname);
  cfg.attach_mode = XDP_MODE_SKB;
  cfg.xdp_flags = 0;
  cfg.xsk_bind_flags = 0;
  cfg.xsk_if_queue = 0;
  cfg.verbose = true;

  signal(SIGINT, exit_application);

  if (!cfg.ifindex) {
    fprintf(stderr, "ERROR: get index from interface failed: '%s'\n", strerror(cfg.ifindex));
    exit(EXIT_FAILURE);
  }

  prog = xdp_program__open_file("agg_xdp.o", "xdp/aggregator", NULL);
  err = libxdp_get_error(prog);
  if (err) {
    libxdp_strerror(err, errmsg, sizeof(errmsg));
    fprintf(stderr, "ERROR: loading program: %s\n", errmsg);
    return err;
  }

  err = xdp_program__attach(prog, cfg.ifindex, cfg.attach_mode, 0);
  if (err) {
    libxdp_strerror(err, errmsg, sizeof(errmsg));
    fprintf(stderr, "Couldn't attach XDP program on iface '%s' : %s (%d)\n", cfg.ifname, errmsg, err);
    return err;
  }

  bpf_obj = xdp_program__bpf_obj(prog);
  map = bpf_object__find_map_by_name(bpf_obj, "xsks_map");
  xsk_map_fd = bpf_map__fd(map);
  if (xsk_map_fd < 0) {
    fprintf(stderr, "ERROR: no xsks map found: %s\n", strerror(xsk_map_fd));
    exit(EXIT_FAILURE);
  }

  if (setrlimit(RLIMIT_MEMLOCK, &rlim)) {
    fprintf(stderr, "ERROR: setrlimit(RLIMIT_MEMLOCK) '%s'\n", strerror(errno));
    exit(EXIT_FAILURE);
  }

  packet_buffer_size = NUM_FRAMES * FRAME_SIZE;
  if (posix_memalign(&packet_buffer, getpagesize(), packet_buffer_size)) {
    fprintf(stderr, "ERROR: Can't allocate buffer memory '%s'\n", strerror(errno));
    exit(EXIT_FAILURE);
  }

  umem = configure_xsk_umem(packet_buffer, packet_buffer_size);
  if (umem == NULL) {
    fprintf(stderr, "ERROR: Can't create umem '%s'\n", strerror(errno));
    exit(EXIT_FAILURE);
  }

  xsk_socket = xsk_configure_socket(&cfg, umem);
  if (xsk_socket == NULL) {
    fprintf(stderr, "ERROR: Can't setup AF_XDP socket '%s'\n", strerror(errno));
    exit(EXIT_FAILURE);
  }

  if (cfg.verbose) {
    ret = pthread_create(&stats_poll_thread, NULL, stats_poll, xsk_socket);
    if (ret) {
      fprintf(stderr, "ERROR: Failed creating statistics thread '%s'\n", strerror(errno));
      exit(EXIT_FAILURE);
    }
  }

  while (!global_exit) {
    recvxdp(xsk_socket);
    //usleep(1000);
  }

  xsk_socket__delete(xsk_socket->xsk);
  xsk_umem__delete(umem->umem);

  return 0;
}
