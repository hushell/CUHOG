//
//    This file is derived from groundHOG, which is used for time counting, 
//    see below for more detail:
//
//    Copyright (c) 2009-2011
//      Patrick Sudowe	<sudowe@umic.rwth-aachen.de>
//      RWTH Aachen University, Germany
//
//    This file is part of groundHOG.
//
//    GroundHOG is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    GroundHOG is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with groundHOG.  If not, see <http://www.gnu.org/licenses/>.
//

#include <time.h>
#include "timer.h"

void startTimer(Timer* t)
{
	clock_gettime(CLOCK_MONOTONIC, &((*t).start));
}

void stopTimer(Timer* t)
{
	clock_gettime(CLOCK_MONOTONIC, &((*t).end));
}

// return timer count in ms
double getTimerValue(Timer* t)
{
	timespec temp;
	if (((*t).end.tv_nsec - (*t).start.tv_nsec) < 0)
		temp.tv_nsec = (*t).end.tv_nsec - (*t).start.tv_nsec + 1000000000;
	else
		temp.tv_nsec = (*t).end.tv_nsec - (*t).start.tv_nsec;
	return temp.tv_nsec / 1000000.;
}

