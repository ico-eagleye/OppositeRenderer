# pragma once

#include <stdio.h>
#include <cstdarg>
#include "config.h"


#if ENABLE_RENDER_DEBUG_OUTPUT
void dbgPrintf(const char * format, ... ) 
{ 
    va_list args;
    va_start(args, format);
    vfprintf(stdout, format, args);
};
#else
inline void dbgPrintf(const char * format, ... ) {};
#endif