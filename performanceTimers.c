/// \file
/// Performance timer functions.
///
/// Use the timer functionality to collect timing and number of calls
/// information for chosen computations (such as force calls) and
/// communication (such as sends, receives, reductions).  Timing results
/// are reported at the end of the run showing overall timings and
/// statistics of timings across ranks.
///
/// A new timer can be added as follows:
/// -# add new handle to the TimerHandle in performanceTimers.h
/// -# provide a corresponding name in timerName
///
/// Note that the order of the handles and names must be the
/// same. This order also determines the order in which the timers are
/// printed. Names can contain leading spaces to show a hierarchical
/// ordering.  Timers with zero calls are omitted from the report.
///
/// Raw timer data is obtained from the getTime() and getTick()
/// functions.  The supplied portable versions of these functions can be
/// replaced with platform specific versions for improved accuracy or
/// lower latency.
/// \see TimerHandle
/// \see getTime
/// \see getTick
///


#include "performanceTimers.h"

#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

#include "performanceTimers.h"
#include "mytype.h"
#include "parallel.h"
#include "yamlOutput.h"

#include <omp.h>

static uint64_t getTime(void);
static double getTick(void);
static void timerStats(void);

/// You must add timer name in same order as enum in .h file.
/// Leading spaces can be specified to show a hierarchy of timers.
char* timerName[numberOfTimers] = {
   "total",
   "Task Creation",
   "init" ,
   "  temp1",
   "  vcm1",
   "  vcm2", 
   "  vcm3", 
   "  vcm4", 
   "  temp2",
   "  random init",
   "loop",
   "timestep",
   "  position",
   "  velocity",
   "  vel+Pos",
   "  redistribute",
   "  redist copy",
   "  force",
   "  ePot reduce",
   "  KE", 
   "  KE reduce", 
   "  Printing", 
   "  ompReduce",
   "halo timer",
   "eam timer"
};

/// Timer data collected.  Also facilitates computing averages and
/// statistics.
typedef struct TimersSt
{
   uint64_t start;     //!< call start time
   uint64_t total;     //!< current total time
   uint64_t count;     //!< current call count
   uint64_t elapsed;   //!< lap time
 
   int minRank;        //!< rank with min value
   int maxRank;        //!< rank with max value

   double minValue;    //!< min over ranks
   double maxValue;    //!< max over ranks
   double average;     //!< average over ranks
   double stdev;       //!< stdev across ranks
} Timers;

Timers **perfTimer;

void initPerfTimers(int numThreads) {
    perfTimer = calloc(numThreads, sizeof(Timers*));
    for(int i=0; i<numThreads; i++) {
        perfTimer[i] = calloc(numberOfTimers, sizeof(Timers));
    }
}


void profileStartThread(int threadNum, const enum TimerHandle handle)
{
    perfTimer[threadNum][handle].start = getTime();
}

void profileStart(const enum TimerHandle handle)
{
    int threadNum = omp_get_thread_num();
    profileStartThread(threadNum, handle);
}

void profileStopThread(int threadNum, const enum TimerHandle handle)
{
    perfTimer[threadNum][handle].count += 1;
    uint64_t delta = getTime() - perfTimer[threadNum][handle].start;
    perfTimer[threadNum][handle].total += delta;
    perfTimer[threadNum][handle].elapsed += delta;
}

void profileStop(const enum TimerHandle handle)
{
    int threadNum = omp_get_thread_num();
    profileStopThread(threadNum, handle);
}

/// \details
/// Return elapsed time (in seconds) since last call with this handle
/// and clear for next lap.

double getElapsedTimeThread(int threadNum, const enum TimerHandle handle)
{
    double etime = getTick() * (double)perfTimer[threadNum][handle].elapsed;
    perfTimer[threadNum][handle].elapsed = 0;
    return etime;
}

double getElapsedTime(const enum TimerHandle handle)
{
    int threadNum = omp_get_thread_num();
    return getElapsedTimeThread(threadNum, handle);
}

/// \details
/// The report contains two blocks.  The upper block is performance
/// information for the printRank.  The lower block is statistical
/// information over all ranks.
void printPerformanceResults(int nGlobalAtoms)
{
    // Collect timer statistics overall and across ranks and threads
    timerStats();

    if (!printRank())
        return;

    // only print timers with non-zero values.
    double tick = getTick();
    double totalTime = perfTimer[0][totalTimer].total*tick;
    uint64_t taskTimeTotal = 0;
    uint64_t totalTasks = 0;

    fprintf(screenOut, "\n\nTimings for Rank %d\n", getMyRank());
    fprintf(screenOut, "        Timer        # Calls    Avg/Call (ms)      Total (s)    %% Total\n");
    fprintf(screenOut, "___________________________________________________________________\n");
    for (int ii=0; ii<numberOfTimers; ++ii) {
        if (perfTimer[0][ii].count > 0) {
            double counterTime = perfTimer[0][ii].total*tick;
            double percentTotal = counterTime/totalTime*100.0;
            if(ii != totalTimer && ii != loopTimer && ii != initTimer) {
                percentTotal = counterTime/(totalTime * omp_get_num_threads()) * 100.0;
            }
            fprintf(screenOut, "%-16s%12"PRIu64"     %11.4f      %8.4f    %8.2f\n", 
                    timerName[ii],
                    perfTimer[0][ii].count,
                    1000*counterTime/(double)perfTimer[0][ii].count,
                    counterTime,
                    percentTotal);
                    //counterTime/loopTime*100.0);
            if(ii != loopTimer && ii != totalTimer && ii != timestepTimer && ii != taskCreationTimer) {
                taskTimeTotal += perfTimer[0][ii].total;
                totalTasks += perfTimer[0][ii].count;
            }
        }
    }

    double totalParTime =  perfTimer[0][totalTimer].total * tick * omp_get_num_threads();
    printf("total number of tasks = %lu\n", totalTasks);
    printf("total time = %f\n", totalParTime);
    printf("total task time = %f\n", taskTimeTotal*tick);
    printf("overhead = %f\n", totalParTime - (taskTimeTotal*tick));

    if(getNRanks() > 1) {
        fprintf(screenOut, "\nTiming Statistics Across %d Ranks:\n", getNRanks());
        fprintf(screenOut, "        Timer        Rank: Min(s)       Rank: Max(s)      Avg(s)    Stdev(s)\n");
        fprintf(screenOut, "_____________________________________________________________________________\n");

        for (int ii = 0; ii < numberOfTimers; ++ii) {
            if(perfTimer[0][ii].count > 0)
                fprintf(screenOut, "%-16s%6d:%10.4f  %6d:%10.4f  %10.4f  %10.4f\n", 
                        timerName[ii], 
                        perfTimer[0][ii].minRank, perfTimer[0][ii].minValue*tick,
                        perfTimer[0][ii].maxRank, perfTimer[0][ii].maxValue*tick,
                        perfTimer[0][ii].average*tick, perfTimer[0][ii].stdev*tick);
        }
        real_t atomsPerTask = nGlobalAtoms/(real_t)getNRanks();
        real_t atomRate = perfTimer[0][computeForceTimer].average * tick * 1e6 /
            (atomsPerTask * perfTimer[0][computeForceTimer].count);
        fprintf(screenOut, "\n---------------------------------------------------\n");
        fprintf(screenOut, " Average atom update rate: %6.2f us/atom/task\n", atomRate);
        fprintf(screenOut, "---------------------------------------------------\n\n");
    }
}

void printPerformanceResultsYaml(FILE* file)
{
   if (! printRank())
      return;

   double tick = getTick();
   double loopTime = perfTimer[0][loopTimer].total*tick;

   fprintf(file,"\nPerformance Results:\n");
   fprintf(file, "  TotalRanks: %d\n", getNRanks());
   fprintf(file, "  ReportingTimeUnits: seconds\n");
   fprintf(file, "Performance Results For Rank %d:\n", getMyRank());
   for (int ii = 0; ii < numberOfTimers; ii++)
   {
      if (perfTimer[0][ii].count > 0)
      {
         double totalTime = perfTimer[0][ii].total*tick;
         fprintf(file, "  Timer: %s\n", timerName[ii]);
         fprintf(file, "    CallCount: %"PRIu64"\n", perfTimer[0][ii].count); 
         fprintf(file, "    AvgPerCall: %8.4f\n", totalTime/(double)perfTimer[0][ii].count);
         fprintf(file, "    Total: %8.4f\n", totalTime);
         fprintf(file, "    PercentLoop: %8.2f\n", totalTime/loopTime*100);
      }
   }

   fprintf(file, "Performance Results Across Ranks:\n");
   for (int ii = 0; ii < numberOfTimers; ii++)
   {
      if (perfTimer[0][ii].count > 0)
      {
         fprintf(file, "  Timer: %s\n", timerName[ii]);
         fprintf(file, "    MinRank: %d\n", perfTimer[0][ii].minRank);
         fprintf(file, "    MinTime: %8.4f\n", perfTimer[0][ii].minValue*tick);     
         fprintf(file, "    MaxRank: %d\n", perfTimer[0][ii].maxRank);
         fprintf(file, "    MaxTime: %8.4f\n", perfTimer[0][ii].maxValue*tick);
         fprintf(file, "    AvgTime: %8.4f\n", perfTimer[0][ii].average*tick);
         fprintf(file, "    StdevTime: %8.4f\n", perfTimer[0][ii].stdev*tick);
      }
   }

   fprintf(file, "\n");
}

/// Returns current time as a 64-bit integer.  This portable version
/// returns the number of microseconds since mindight, Jamuary 1, 1970.
/// Hence, timing data will have a resolution of 1 microsecond.
/// Platforms with access to calls with lower latency or higher
/// resolution (such as a cycle counter) may wish to replace this
/// implementation and change the conversion factor in getTick as
/// appropriate.
/// \see getTick for the conversion factor between the integer time
/// units of this function and seconds.
static uint64_t getTime(void)
{
   struct timeval ptime;
   uint64_t t = 0;
   gettimeofday(&ptime, (struct timezone *)NULL);
   t = ((uint64_t)1000000)*(uint64_t)ptime.tv_sec + (uint64_t)ptime.tv_usec;

   return t; 
}

/// Returns the factor for converting the integer time reported by
/// getTime into seconds.  The portable getTime returns values in units
/// of microseconds so the conversion is simply 1e-6.
/// \see getTime
static double getTick(void)
{
   double seconds_per_cycle = 1.0e-6;
   return seconds_per_cycle; 
}

void reducePerfTimers() {
    int numThreads = omp_get_num_threads();
    for(int i=1; i<numThreads; i++) {
        for(int timer=0; timer < numberOfTimers; timer++) {
            perfTimer[0][timer].total   += perfTimer[i][timer].total;
            perfTimer[0][timer].count   += perfTimer[i][timer].count;
            perfTimer[0][timer].elapsed += perfTimer[i][timer].elapsed;
        }
    }
}

/// Collect timer statistics across ranks.
void timerStats(void)
{
   double sendBuf[numberOfTimers], recvBuf[numberOfTimers];
   reducePerfTimers();
   
   // Determine average of each timer across ranks
   for (int ii = 0; ii < numberOfTimers; ii++)
      sendBuf[ii] = (double)perfTimer[0][ii].total;
   addDoubleParallel(sendBuf, recvBuf, numberOfTimers);

   for (int ii = 0; ii < numberOfTimers; ii++)
      perfTimer[0][ii].average = recvBuf[ii] / (double)getNRanks();


   // Determine min and max across ranks and which rank
   RankReduceData reduceSendBuf[numberOfTimers], reduceRecvBuf[numberOfTimers];
   for (int ii = 0; ii < numberOfTimers; ii++)
   {
      reduceSendBuf[ii].val = (double)perfTimer[0][ii].total;
      reduceSendBuf[ii].rank = getMyRank();
   }
   minRankDoubleParallel(reduceSendBuf, reduceRecvBuf, numberOfTimers);   
   for (int ii = 0; ii < numberOfTimers; ii++)
   {
      perfTimer[0][ii].minValue = reduceRecvBuf[ii].val;
      perfTimer[0][ii].minRank = reduceRecvBuf[ii].rank;
   }
   maxRankDoubleParallel(reduceSendBuf, reduceRecvBuf, numberOfTimers);   
   for (int ii = 0; ii < numberOfTimers; ii++)
   {
      perfTimer[0][ii].maxValue = reduceRecvBuf[ii].val;
      perfTimer[0][ii].maxRank = reduceRecvBuf[ii].rank;
   }
   
   // Determine standard deviation
   for (int ii = 0; ii < numberOfTimers; ii++)
   {
      double temp = (double)perfTimer[0][ii].total - perfTimer[0][ii].average;
      sendBuf[ii] = temp * temp;
   }
   addDoubleParallel(sendBuf, recvBuf, numberOfTimers);
   for (int ii = 0; ii < numberOfTimers; ii++)
   {
      perfTimer[0][ii].stdev = sqrt(recvBuf[ii] / (double) getNRanks());
   }
}

