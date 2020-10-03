#pragma once

//#define MAX_DWELL 256
#define BS 256

/** gets the color, given the dwell (on host) */
#define CUT_DWELL (MAX_DWELL / 4)


#define MAX_DWELL 256
/** block size along */
#define BSX 64
#define BSY 4
/** maximum recursion depth */
#define MAX_DEPTH 4
/** region below which do per-pixel */
#define MIN_SIZE 32
/** subdivision factor along each axis */
#define SUBDIV 4
/** subdivision when launched from host */
#define INIT_SUBDIV 32