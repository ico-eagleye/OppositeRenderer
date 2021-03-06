/* 
 * Copyright (c) 2014 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 *
 * Contributions: Valdis Vilcans
*/

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
    fflush(stdout);
};
#else
inline void dbgPrintf(const char * format, ... ) {};
#endif