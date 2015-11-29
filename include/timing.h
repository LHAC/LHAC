//
//  timing.h
//  pepper
//
//  Created by Xiaocheng Tang on 3/21/14.
//  Copyright (c) 2014 Xiaocheng Tang. All rights reserved.
//

#ifndef pepper_timing_h
#define pepper_timing_h

#include <iostream>


#if defined(_MSC_VER) || defined(_WIN32) || defined(WINDOWS)

#include <time.h>
#include <windows.h>
#define random rand
#define srandom srand

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif


struct timezone
{
    int  tz_minuteswest; /* minutes W of Greenwich */
    int  tz_dsttime;     /* type of dst correction */
};

int gettimeofday(struct timeval *tv, struct timezone *tz)
{
    FILETIME ft;
    unsigned __int64 tmpres = 0;
    static int tzflag = 0;
    
    if (NULL != tv)
    {
        GetSystemTimeAsFileTime(&ft);
        
        tmpres |= ft.dwHighDateTime;
        tmpres <<= 32;
        tmpres |= ft.dwLowDateTime;
        
        tmpres /= 10;  /*convert into microseconds*/
        /*converting file time to unix epoch*/
        tmpres -= DELTA_EPOCH_IN_MICROSECS;
        tv->tv_sec = (long)(tmpres / 1000000UL);
        tv->tv_usec = (long)(tmpres % 1000000UL);
    }
    
    if (NULL != tz)
    {
        if (!tzflag)
        {
            _tzset();
            tzflag++;
        }
        tz->tz_minuteswest = _timezone / 60;
        tz->tz_dsttime = _daylight;
    }
    
    return 0;
}

#else
#include <sys/time.h>
#endif


using namespace std;

/// Class Timer
class Timer {
public:
    /// Empty constructor
    Timer();
    /// Destructor
    ~Timer();
    
    /// start the time
    void inline start() { _running=true;
        gettimeofday(_time1,NULL); };
    /// stop the time
    void inline stop() {
        gettimeofday(_time2,NULL);
        _running=false;
        _cumul+=  static_cast<double>((_time2->tv_sec - (_time1->tv_sec))*1000000 + _time2->tv_usec-_time1->tv_usec)/1000000.0;
    };
    /// reset the timer
    void inline reset() { _cumul=0;
        gettimeofday(_time1,NULL); };
    /// print the elapsed time
    void inline printElapsed();
    /// print the elapsed time
    double inline getElapsed() const;
    
private:
    struct timeval* _time1;
    struct timeval* _time2;
    bool _running;
    double _cumul;
};

/// Constructor
Timer::Timer() :_running(false) ,_cumul(0) {
    _time1 = (struct timeval*)malloc(sizeof(struct timeval));
    _time2 = (struct timeval*)malloc(sizeof(struct timeval));
};

/// Destructor
Timer::~Timer() {
    free(_time1);
    free(_time2);
}

/// print the elapsed time
inline void Timer::printElapsed() {
    if (_running) {
        gettimeofday(_time2,NULL);
        cerr << "Time elapsed : " << _cumul + static_cast<double>((_time2->tv_sec -
                                                                   _time1->tv_sec)*1000000 + _time2->tv_usec-_time1->tv_usec)/1000000.0 << endl;
    } else {
        cerr << "Time elapsed : " << _cumul << endl;
    }
};

/// print the elapsed time
double inline Timer::getElapsed() const {
    if (_running) {
        gettimeofday(_time2,NULL);
        return _cumul +
        static_cast<double>((_time2->tv_sec -
                             _time1->tv_sec)*1000000 + _time2->tv_usec-
                            _time1->tv_usec)/1000000.0;
    } else {
        return _cumul;
    }
}

#ifdef USE_ACCELERATE
#include <CoreFoundation/CoreFoundation.h>
#else // unix/linux
//#include <time.h>
//#include <stdint.h>

/**
 *  The Linux clock_gettime is reasonably fast, has good resolution, and is not
 *  affected by TurboBoost.  Using MONOTONIC_RAW also means that the timer is
 *  not subject to NTP adjustments, which is preferably since an adjustment in
 *  mid-experiment could produce some funky results.
 */

//#define BILLION 1E9

inline double CFAbsoluteTimeGetCurrent()
{
//    struct timespec t;
//    clock_gettime(CLOCK_REALTIME, &t);
//    double tt = ((double)t.tv_sec) + ((double)t.tv_nsec) / BILLION;
//    return tt;
    static struct timeval t;
    gettimeofday(&t,NULL);
    return static_cast<double>(t.tv_sec*1000000 + t.tv_usec)/1000000.0;
}

#endif


#endif
