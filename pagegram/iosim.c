// Flash cost of an n-gram table: blocks read per token, LRU page-cache misses, direct-read throughput.
// Addressing matches pagegram/memory.py.
//
//   iosim sim   TOKENS N V LAYOUT K S P2 P3 SIZES(csv) ROWS_PER_PAGE CACHE_PAGES(csv)
//   iosim bench TOKENS N V LAYOUT K S P2 P3 SIZES(csv) ROWS_PER_PAGE FILE THREADS
//   iosim pages TOKENS N V LAYOUT K S P2 P3 SIZES(csv) ROWS_PER_PAGE   (distinct page ids per token)
//
// paged:  a token reads the block of page (b mod P2) and of page P2 + (c mod P3); here ROWS_PER_PAGE is
//         the number of pages per 4 KiB block (pages never straddle blocks).
// global: a token reads the blocks of its 2K head-rows, ROWS_PER_PAGE head-rows per block.
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define EOS 2
static const int64_t M[7] = {0x5BD1E995, 0x27D4EB2F, 0x165667B1, 0x61C88647, 0x2545F491, 0x3C6EF372, 0x4F1BBCDD};
static const int64_t A[2][8][3] = {
    {{0x7f5885ed, 0x48ecdc53, 0x47183b11}, {0x468f9463, 0x559e0b89, 0x4dd332af},
     {0x4d6a04e7, 0x719fe317, 0x51d40e7d}, {0x7ac5a7ef, 0x438ce42b, 0x6b2f068f},
     {0x6bc310ef, 0x6349370d, 0x5a896ed7}, {0x65c10571, 0x7c01717d, 0x7f57b517},
     {0x413d6137, 0x50669347, 0x4a29bb65}, {0x64575d93, 0x79887bcf, 0x6c96af2f}},
    {{0x49ad3517, 0x78574a7f, 0x7b41221d}, {0x6d6f659b, 0x786913ef, 0x61d66543},
     {0x571f7899, 0x6c284c17, 0x6b6f07c3}, {0x6392b4ad, 0x7b05c7bb, 0x4f91118b},
     {0x5fde71d7, 0x5170f3f9, 0x5ca28ab7}, {0x7b51ee09, 0x42ba98f7, 0x4c06b121},
     {0x61538671, 0x77c73d95, 0x5fe57185}, {0x48f95c27, 0x5737c629, 0x4b3e4c63}}};

typedef struct {
    int paged, K;
    int64_t S, P2, P3, rows_per_page, sizes[16], bases[16];
} Geo;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int parse_csv(const char *s, int64_t *out, int max) {
    int n = 0;
    while (*s && n < max) {
        out[n++] = strtoll(s, (char **)&s, 10);
        if (*s == ',') s++;
    }
    return n;
}

// Distinct block ids read by token t, written to pg; returns the count.
static int token_pages(const uint16_t *tok, int64_t t, int64_t V, const Geo *g, int64_t *pg) {
    int64_t x = tok[t], p1 = t >= 1 ? tok[t - 1] : V, p2 = t >= 2 ? tok[t - 2] : V;
    int start = x == EOS;
    if (start || p1 == EOS) p2 = V;
    if (start) p1 = V;
    int64_t b = (p1 * M[0]) ^ (x * M[1]);
    int64_t c = (p2 * M[2]) ^ (p1 * M[3]) ^ (x * M[4]);
    int n = 0;
    if (g->paged) {
        int64_t ppb = g->rows_per_page, blocks2 = (g->P2 + ppb - 1) / ppb;
        pg[n++] = (b % g->P2) / ppb;
        pg[n++] = blocks2 + (c % g->P3) / ppb;
        return n;
    }
    for (int k = 0; k < g->K; k++) {
        int64_t u2 = (p1 * A[0][k][0]) ^ (x * A[0][k][1]);
        int64_t u3 = (p2 * A[1][k][0]) ^ (p1 * A[1][k][1]) ^ (x * A[1][k][2]);
        int64_t r2 = g->bases[k] + u2 % g->sizes[k];
        int64_t r3 = g->bases[g->K + k] + u3 % g->sizes[g->K + k];
        int64_t q[2] = {r2 / g->rows_per_page, r3 / g->rows_per_page};
        for (int j = 0; j < 2; j++) {
            int dup = 0;
            for (int i = 0; i < n; i++) dup |= pg[i] == q[j];
            if (!dup) pg[n++] = q[j];
        }
    }
    return n;
}

static int64_t total_pages(const Geo *g) {
    if (g->paged) return (g->P2 + g->rows_per_page - 1) / g->rows_per_page + (g->P3 + g->rows_per_page - 1) / g->rows_per_page;
    return (g->bases[2 * g->K - 1] + g->sizes[2 * g->K - 1]) / g->rows_per_page + 1;
}

typedef struct {
    int64_t cap, size, head, tail, misses;
    int32_t *map, *prev, *next;
    int64_t *page;
} LRU;

static void lru_init(LRU *c, int64_t cap, int64_t pages) {
    memset(c, 0, sizeof *c);
    c->cap = cap;
    c->head = c->tail = -1;
    c->map = malloc(pages * sizeof(int32_t));
    memset(c->map, 0xff, pages * sizeof(int32_t));
    c->prev = malloc(cap * sizeof(int32_t));
    c->next = malloc(cap * sizeof(int32_t));
    c->page = malloc(cap * sizeof(int64_t));
}

static void lru_unlink(LRU *c, int32_t i) {
    if (c->prev[i] >= 0) c->next[c->prev[i]] = c->next[i]; else c->head = c->next[i];
    if (c->next[i] >= 0) c->prev[c->next[i]] = c->prev[i]; else c->tail = c->prev[i];
}

static void lru_front(LRU *c, int32_t i) {
    c->prev[i] = -1;
    c->next[i] = (int32_t)c->head;
    if (c->head >= 0) c->prev[c->head] = i;
    c->head = i;
    if (c->tail < 0) c->tail = i;
}

static void lru_access(LRU *c, int64_t p) {
    int32_t i = c->map[p];
    if (i >= 0) {
        lru_unlink(c, i);
        lru_front(c, i);
        return;
    }
    c->misses++;
    if (c->cap == 0) return;
    if (c->size < c->cap) {
        i = (int32_t)c->size++;
    } else {
        i = (int32_t)c->tail;
        lru_unlink(c, i);
        c->map[c->page[i]] = -1;
    }
    c->page[i] = p;
    c->map[p] = i;
    lru_front(c, i);
}

static uint16_t *load_tokens(const char *path, int64_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    int64_t len = ftell(f) / 2;
    if (*n <= 0 || *n > len) *n = len;
    fseek(f, 0, SEEK_SET);
    uint16_t *t = malloc(*n * 2);
    if (fread(t, 2, *n, f) != (size_t)*n) { perror("read"); exit(1); }
    fclose(f);
    return t;
}

typedef struct {
    int fd, tid, nthreads;
    int64_t *offs, n_offs;
    char *buf;
    double lat_sum;
    int64_t done;
} Worker;

static void *worker(void *arg) {
    Worker *w = arg;
    for (int64_t i = w->tid; i < w->n_offs; i += w->nthreads) {
        double t0 = now();
        if (pread(w->fd, w->buf, 4096, w->offs[i]) != 4096) { perror("pread"); exit(1); }
        w->lat_sum += now() - t0;
        w->done++;
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 12 + (strcmp(argv[1], "pages") != 0)) {
        fprintf(stderr, "usage: see the header of iosim.c\n");
        return 2;
    }
    const char *mode = argv[1];
    int64_t n = strtoll(argv[3], 0, 10), V = strtoll(argv[4], 0, 10);
    Geo g = {0};
    g.paged = strcmp(argv[5], "paged") == 0;
    g.K = atoi(argv[6]);
    g.S = strtoll(argv[7], 0, 10);
    g.P2 = strtoll(argv[8], 0, 10);
    g.P3 = strtoll(argv[9], 0, 10);
    int ns = parse_csv(argv[10], g.sizes, 16);
    g.rows_per_page = strtoll(argv[11], 0, 10);
    if (!g.paged && ns != 2 * g.K) { fprintf(stderr, "need 2K sizes\n"); return 2; }
    int64_t acc = 0;
    for (int i = 0; i < ns; i++) { g.bases[i] = acc; acc += g.sizes[i]; }
    uint16_t *tok = load_tokens(argv[2], &n);
    int64_t pages = total_pages(&g), pg[16];

    if (strcmp(mode, "pages") == 0) {
        for (int64_t t = 0; t < n; t++) {
            int k = token_pages(tok, t, V, &g, pg);
            for (int j = 0; j < k; j++) printf("%s%lld", j ? " " : "", (long long)pg[j]);
            printf("\n");
        }
        return 0;
    }

    if (strcmp(mode, "sim") == 0) {
        int64_t caps[16];
        int nc = parse_csv(argv[12], caps, 16);
        LRU c[16];
        for (int i = 0; i < nc; i++) lru_init(&c[i], caps[i], pages);
        int64_t reads = 0;
        double t0 = now();
        for (int64_t t = 0; t < n; t++) {
            int k = token_pages(tok, t, V, &g, pg);
            reads += k;
            for (int i = 0; i < nc; i++)
                for (int j = 0; j < k; j++) lru_access(&c[i], pg[j]);
        }
        printf("{\"layout\": \"%s\", \"tokens\": %lld, \"pages\": %lld, \"pages_per_token\": %.4f, \"seconds\": %.2f, \"cache\": [",
               g.paged ? "paged" : "global", (long long)n, (long long)pages, (double)reads / n, now() - t0);
        for (int i = 0; i < nc; i++)
            printf("%s{\"pages\": %lld, \"misses_per_token\": %.5f}", i ? ", " : "", (long long)caps[i],
                   (double)c[i].misses / n);
        printf("]}\n");
        return 0;
    }

    if (strcmp(mode, "bench") == 0) {
        const char *path = argv[12];
        int threads = argc > 13 ? atoi(argv[13]) : 32;
        int fd = open(path, O_RDONLY | O_DIRECT);
        int direct = fd >= 0;
        if (!direct) {
            fd = open(path, O_RDONLY);
            if (fd < 0) { perror(path); return 1; }
        }
        struct stat st;
        fstat(fd, &st);
        if ((int64_t)st.st_size < pages * 4096) { fprintf(stderr, "file smaller than the table\n"); return 1; }
        posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        int64_t *offs = malloc(n * 16 * sizeof(int64_t)), no = 0;
        for (int64_t t = 0; t < n; t++) {
            int k = token_pages(tok, t, V, &g, pg);
            for (int j = 0; j < k; j++) offs[no++] = pg[j] * 4096;
        }
        Worker w[256];
        pthread_t th[256];
        if (threads > 256) threads = 256;
        double t0 = now();
        for (int i = 0; i < threads; i++) {
            w[i] = (Worker){fd, i, threads, offs, no, NULL, 0, 0};
            if (posix_memalign((void **)&w[i].buf, 4096, 4096)) return 1;
            pthread_create(&th[i], NULL, worker, &w[i]);
        }
        double lat = 0;
        for (int i = 0; i < threads; i++) {
            pthread_join(th[i], NULL);
            lat += w[i].lat_sum;
        }
        double dt = now() - t0;
        printf("{\"layout\": \"%s\", \"direct\": %s, \"threads\": %d, \"tokens\": %lld, \"reads\": %lld, "
               "\"seconds\": %.3f, \"reads_per_s\": %.0f, \"tokens_per_s\": %.1f, \"mean_read_latency_us\": %.1f}\n",
               g.paged ? "paged" : "global", direct ? "true" : "false", threads, (long long)n, (long long)no, dt,
               no / dt, n / dt, 1e6 * lat / no);
        return 0;
    }
    fprintf(stderr, "unknown mode %s\n", mode);
    return 2;
}
