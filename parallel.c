/// \file
/// Wrappers for MPI functions.  This should be the only compilation 
/// unit in the code that directly calls MPI functions.  To build a pure
/// serial version of the code with no MPI, do not define DO_MPI.  If
/// DO_MPI is not defined then all MPI functionality is replaced with
/// equivalent single task behavior.

#include "parallel.h"
#include "performanceTimers.h"

#ifdef DO_MPI
#include <mpi.h>
#endif

#include <stdio.h>
#include <time.h>
#include <string.h>
#include <assert.h>

static int myRank = 0;
static int nRanks = 1;



//-------- Int Reductions --------------

void reduceInt(int *depArray, int arraySize, int innerStride)
{
    int numDeps = 16;
    int cellsPerTask = numDeps * innerStride;
    while( innerStride < arraySize ) { 
        for(int iBox=0; iBox < arraySize; iBox +=cellsPerTask) {
#pragma omp task depend(inout: depArray[iBox]) \
                 depend(in   : depArray[iBox+   innerStride],\
                               depArray[iBox+2 *innerStride], depArray[iBox+3 *innerStride],\
                               depArray[iBox+4 *innerStride], depArray[iBox+5 *innerStride],\
                               depArray[iBox+6 *innerStride], depArray[iBox+7 *innerStride],\
                               depArray[iBox+8 *innerStride], depArray[iBox+9 *innerStride],\
                               depArray[iBox+10*innerStride], depArray[iBox+11*innerStride],\
                               depArray[iBox+12*innerStride], depArray[iBox+13*innerStride],\
                               depArray[iBox+14*innerStride], depArray[iBox+15*innerStride])
            {
                startTimer(ompReduceTimer);
                for(int i=iBox+innerStride; i<iBox+cellsPerTask && i<arraySize; i+=innerStride) {
                    depArray[iBox] += depArray[i];
                    depArray[i] = 0.;
                }
                stopTimer(ompReduceTimer);
            }
        }
        innerStride = cellsPerTask;
        cellsPerTask *= numDeps;
    }
}

void reduceRowInt(int *depArrayRow, int rowSize) {
    for(int i=1; i<rowSize; i++) {
        depArrayRow[0] += depArrayRow[i];
        depArrayRow[i] = 0;
    }
}

void ompReduceRowInt(int *depArray, int gridSize[3])
{
    for(int z=0; z < gridSize[2]; z++) {
        for(int y=0; y < gridSize[1]; y++) {
            int rowBox = z*gridSize[1]*gridSize[0] + y*gridSize[0];
#pragma omp task depend(inout: depArray[rowBox])
            {
                startTimer(ompReduceTimer);
                reduceRowInt(&depArray[rowBox], gridSize[0]);
                stopTimer(ompReduceTimer);
            }
        }
    }
    reduceInt(depArray, gridSize[0]*gridSize[1]*gridSize[2], gridSize[0]);
}

//-------- R3 Reductions --------------

void reduceR3(real3 *depArray, int arraySize, int innerStride)
{
    int numDeps = 16;
    int cellsPerTask = numDeps * innerStride;
    while( innerStride < arraySize ) { 
        for(int iBox=0; iBox < arraySize; iBox +=cellsPerTask) {
#pragma omp task depend(inout: depArray[iBox]) \
                 depend(in   : depArray[iBox+   innerStride],\
                               depArray[iBox+2 *innerStride], depArray[iBox+3 *innerStride],\
                               depArray[iBox+4 *innerStride], depArray[iBox+5 *innerStride],\
                               depArray[iBox+6 *innerStride], depArray[iBox+7 *innerStride],\
                               depArray[iBox+8 *innerStride], depArray[iBox+9 *innerStride],\
                               depArray[iBox+10*innerStride], depArray[iBox+11*innerStride],\
                               depArray[iBox+12*innerStride], depArray[iBox+13*innerStride],\
                               depArray[iBox+14*innerStride], depArray[iBox+15*innerStride])
            {
                startTimer(ompReduceTimer);
                for(int i=iBox+innerStride; i<iBox+cellsPerTask && i<arraySize; i+=innerStride) {
                    for(int j=0; j<3; j++) {
                        depArray[iBox][j] += depArray[i][j];
                        depArray[i][j] = 0.;
                    }
                }
                stopTimer(ompReduceTimer);
            }
        }
        innerStride = cellsPerTask;
        cellsPerTask *= numDeps;
    }
}

void reduceRowR3(real3 *depArrayRow, int rowSize) {
    for(int i=1; i<rowSize; i++) {
        for(int j=0; j<3; j++) {
            depArrayRow[0][j] += depArrayRow[i][j];
            depArrayRow[i][j] = 0;
        }
    }
}

void ompReduceRowR3(real3 *depArray, int gridSize[3]) 
{
    for(int z=0; z < gridSize[2]; z++) {
        for(int y=0; y < gridSize[1]; y++) {
            int rowBox = z*gridSize[1]*gridSize[0] + y*gridSize[0];
#pragma omp task depend(inout: depArray[rowBox])
            {
                startTimer(ompReduceTimer);
                reduceRowR3(&depArray[rowBox], gridSize[0]);
                stopTimer(ompReduceTimer);
            }
        }
    }
    reduceR3(depArray, gridSize[0]*gridSize[1]*gridSize[2], gridSize[0]);
}

//-------- Real Reductions --------------

void reduceReal(real_t *depArray, int arraySize, int innerStride)
{
    int numDeps = 16;
    int cellsPerTask = numDeps * innerStride;
    while( innerStride < arraySize ) { 
        for(int iBox=0; iBox < arraySize; iBox +=cellsPerTask) {
#pragma omp task depend(inout: depArray[iBox]) \
                 depend(in   : depArray[iBox+   innerStride],\
                               depArray[iBox+2 *innerStride], depArray[iBox+3 *innerStride],\
                               depArray[iBox+4 *innerStride], depArray[iBox+5 *innerStride],\
                               depArray[iBox+6 *innerStride], depArray[iBox+7 *innerStride],\
                               depArray[iBox+8 *innerStride], depArray[iBox+9 *innerStride],\
                               depArray[iBox+10*innerStride], depArray[iBox+11*innerStride],\
                               depArray[iBox+12*innerStride], depArray[iBox+13*innerStride],\
                               depArray[iBox+14*innerStride], depArray[iBox+15*innerStride])
            {
                startTimer(ompReduceTimer);
                for(int i=iBox+innerStride; i<iBox+cellsPerTask && i<arraySize; i+=innerStride) {
                    depArray[iBox] += depArray[i];
                    depArray[i] = 0.;
                }
                stopTimer(ompReduceTimer);
            }
        }
        innerStride = cellsPerTask;
        cellsPerTask *= numDeps;
    }
}

void reduceRowReal(real_t *depArrayRow, int rowSize) {
    for(int i=1; i<rowSize; i++) {
        depArrayRow[0] += depArrayRow[i];
        depArrayRow[i] = 0;
    }
}

void ompReduceRowReal(real_t *depArray, int gridSize[3]) 
{
    for(int z=0; z < gridSize[2]; z++) {
        for(int y=0; y < gridSize[1]; y++) {
            int rowBox = z*gridSize[1]*gridSize[0] + y*gridSize[0];
#pragma omp task depend(inout: depArray[rowBox])
            {
                startTimer(ompReduceTimer);
                reduceRowReal(&depArray[rowBox], gridSize[0]);
                stopTimer(ompReduceTimer);
            }
        }
    }
    reduceReal(depArray, gridSize[0]*gridSize[1]*gridSize[2], gridSize[0]);
}

#ifdef DO_MPI
#ifdef SINGLE
#define REAL_MPI_TYPE MPI_FLOAT
#else
#define REAL_MPI_TYPE MPI_DOUBLE
#endif

#endif

int getNRanks()
{
   return nRanks;
}

int getMyRank()   
{
   return myRank;
}

/// \details
/// For now this is just a check for rank 0 but in principle it could be
/// more complex.  It is also possible to suppress practically all
/// output by causing this function to return 0 for all ranks.
int printRank()
{
   if (myRank == 0) return 1;
   return 0;
}

void timestampBarrier(const char* msg)
{
   barrierParallel();
   if (! printRank())
      return;
   time_t t= time(NULL);
   char* timeString = ctime(&t);
   timeString[24] = '\0'; // clobber newline
   fprintf(screenOut, "%s: %s\n", timeString, msg);
   fflush(screenOut);
}

void initParallel(int* argc, char*** argv)
{
#ifdef DO_MPI
   MPI_Init(argc, argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
#endif
}

void destroyParallel()
{
#ifdef DO_MPI
   MPI_Finalize();
#endif
}

void barrierParallel()
{
#ifdef DO_MPI
   MPI_Barrier(MPI_COMM_WORLD);
#endif
}

/// \param [in]  sendBuf Data to send.
/// \param [in]  sendLen Number of bytes to send.
/// \param [in]  dest    Rank in MPI_COMM_WORLD where data will be sent.
/// \param [out] recvBuf Received data.
/// \param [in]  recvLen Maximum number of bytes to receive.
/// \param [in]  source  Rank in MPI_COMM_WORLD from which to receive.
/// \return Number of bytes received.
int sendReceiveParallel(void* sendBuf, int sendLen, int dest,
                        void* recvBuf, int recvLen, int source)
{
#ifdef DO_MPI
   int bytesReceived;
   MPI_Status status;
   MPI_Sendrecv(sendBuf, sendLen, MPI_BYTE, dest,   0,
                recvBuf, recvLen, MPI_BYTE, source, 0,
                MPI_COMM_WORLD, &status);
   MPI_Get_count(&status, MPI_BYTE, &bytesReceived);

   return bytesReceived;
#else
   assert(source == dest);
   memcpy(recvBuf, sendBuf, sendLen);

   return sendLen;
#endif
}

void addIntParallel(int* sendBuf, int* recvBuf, int count)
{
#ifdef DO_MPI
   MPI_Allreduce(sendBuf, recvBuf, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
   for (int ii=0; ii<count; ++ii)
      recvBuf[ii] = sendBuf[ii];
#endif
}

void addRealParallel(real_t* sendBuf, real_t* recvBuf, int count)
{
#ifdef DO_MPI
   MPI_Allreduce(sendBuf, recvBuf, count, REAL_MPI_TYPE, MPI_SUM, MPI_COMM_WORLD);
#else
   for (int ii=0; ii<count; ++ii)
      recvBuf[ii] = sendBuf[ii];
#endif
}

void addDoubleParallel(double* sendBuf, double* recvBuf, int count)
{
#ifdef DO_MPI
   MPI_Allreduce(sendBuf, recvBuf, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
   for (int ii=0; ii<count; ++ii)
      recvBuf[ii] = sendBuf[ii];
#endif
}

void maxIntParallel(int* sendBuf, int* recvBuf, int count)
{
#ifdef DO_MPI
   MPI_Allreduce(sendBuf, recvBuf, count, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#else
   for (int ii=0; ii<count; ++ii)
      recvBuf[ii] = sendBuf[ii];
#endif
}


void minRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count)
{
#ifdef DO_MPI
   MPI_Allreduce(sendBuf, recvBuf, count, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
#else
   for (int ii=0; ii<count; ++ii)
   {
      recvBuf[ii].val = sendBuf[ii].val;
      recvBuf[ii].rank = sendBuf[ii].rank;
   }
#endif
}

void maxRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count)
{
#ifdef DO_MPI
   MPI_Allreduce(sendBuf, recvBuf, count, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
#else
   for (int ii=0; ii<count; ++ii)
   {
      recvBuf[ii].val = sendBuf[ii].val;
      recvBuf[ii].rank = sendBuf[ii].rank;
   }
#endif
}

/// \param [in] count Length of buf in bytes.
void bcastParallel(void* buf, int count, int root)
{
#ifdef DO_MPI
   MPI_Bcast(buf, count, MPI_BYTE, root, MPI_COMM_WORLD);
#endif
}

int builtWithMpi(void)
{
#ifdef DO_MPI
   return 1;
#else
   return 0;
#endif
}


