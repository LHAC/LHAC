//
//  timing.h
//  pepper
//
//  Created by Xiaocheng Tang on 3/21/14.
//  Copyright (c) 2014 Xiaocheng Tang. All rights reserved.
//

#ifndef pepper_timing_h
#define pepper_timing_h

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#else // unix/linux
#include <time.h>
#include <stdint.h>

/**
 *  The Linux clock_gettime is reasonably fast, has good resolution, and is not
 *  affected by TurboBoost.  Using MONOTONIC_RAW also means that the timer is
 *  not subject to NTP adjustments, which is preferably since an adjustment in
 *  mid-experiment could produce some funky results.
 */

#define BILLION 1E9

inline double CFAbsoluteTimeGetCurrent()
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    double tt = ((double)t.tv_sec) + ((double)t.tv_nsec) / BILLION;
    return tt;
}

#endif


#endif
