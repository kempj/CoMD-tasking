/// \file
/// Leapfrog time integrator

#include "timestep.h"

#include <omp.h>

#include "CoMDTypes.h"
#include "linkCells.h"
#include "parallel.h"
#include "performanceTimers.h"

static void advanceVelocity(SimFlat* s, int nBoxes, real_t dt);
static void advancePosition(SimFlat* s, int nBoxes, real_t dt);

extern double *reductionArray;
extern int *reductionArrayInt;
extern double globalEnergy;

/// Advance the simulation time to t+dt using a leap frog method
/// (equivalent to velocity verlet).
///
/// Forces must be computed before calling the integrator the first time.
///
///  - Advance velocities half time step using forces
///  - Advance positions full time step using velocities
///  - Update link cells and exchange remote particles
///  - Compute forces
///  - Update velocities half time step using forces
///
/// This leaves positions, velocities, and forces at t+dt, with the
/// forces ready to perform the half step velocity update at the top of
/// the next call.
///
/// After nSteps the kinetic energy is computed for diagnostic output.
double timestep(SimFlat* s, int nSteps, real_t dt)
{
    for (int ii=0; ii<nSteps; ++ii) {
        advanceVelocity(s, s->boxes->nLocalBoxes, 0.5*dt);//in: atomF, atomP, out: atomP
        advancePosition(s, s->boxes->nLocalBoxes, dt);    //in: atomP, out: atomR
        redistributeAtoms(s);                             //potentially entire atoms moved, but 9->1 deps
        computeForce(s);                                  //in: atomR, out: atomF, atomU, reduction, but 9->1 deps
        advanceVelocity(s, s->boxes->nLocalBoxes, 0.5*dt);//in: atomF, atomP, out: atomP

    }
    kineticEnergy(s);//atomP -> KE and nAtoms -> nLocal
    return s->ePotential;
}

void computeForce(SimFlat* s)
{
    s->pot->force(s);
}

void advanceVelocity(SimFlat* s, int nBoxes, real_t dt)
{
    real3 *atomP = s->atoms->p;
    real3 *atomF = s->atoms->f;
    for(int z=0; z < s->boxes->gridSize[2]; z++) {
        for(int y=0; y < s->boxes->gridSize[1]; y++) {
            int rowBox = z*s->boxes->gridSize[1]*s->boxes->gridSize[0] + y*s->boxes->gridSize[0];
#pragma omp task depend(inout: atomP[rowBox*MAXATOMS]) depend(in: atomF[rowBox*MAXATOMS])
            {
                startTimer(velocityTimer);
                for(int iBox=rowBox; iBox < rowBox + s->boxes->gridSize[0]; iBox++) {
                    for (int iOff=MAXATOMS*iBox,ii=0; ii<s->boxes->nAtoms[iBox]; ii++,iOff++) {
                        s->atoms->p[iOff][0] += dt*s->atoms->f[iOff][0];
                        s->atoms->p[iOff][1] += dt*s->atoms->f[iOff][1];
                        s->atoms->p[iOff][2] += dt*s->atoms->f[iOff][2];
                    }
                }
                stopTimer(velocityTimer);
            }
        }
    }
}

void advancePosition(SimFlat* s, int nBoxes, real_t dt)
{
    real3 *atomP = s->atoms->p;
    real3 *atomR = s->atoms->r;
    for(int z=0; z < s->boxes->gridSize[2]; z++) {
        for(int y=0; y < s->boxes->gridSize[1]; y++) {
            int rowBox = z*s->boxes->gridSize[1]*s->boxes->gridSize[0]+y*s->boxes->gridSize[0];
#pragma omp task depend(inout: atomR[rowBox*MAXATOMS]) depend(in: atomP[rowBox*MAXATOMS])
            {
                startTimer(positionTimer);
                //printf("Position for %d - %d\n", rowBox, rowBox + s->boxes->gridSize[0]);
                for(int iBox=rowBox; iBox < rowBox + s->boxes->gridSize[0]; iBox++) {
                    for (int iOff=MAXATOMS*iBox,ii=0; ii<s->boxes->nAtoms[iBox]; ii++,iOff++) {
                        int iSpecies = s->atoms->iSpecies[iOff];
                        real_t invMass = 1.0/s->species[iSpecies].mass;
                        s->atoms->r[iOff][0] += dt*s->atoms->p[iOff][0]*invMass;
                        s->atoms->r[iOff][1] += dt*s->atoms->p[iOff][1]*invMass;
                        s->atoms->r[iOff][2] += dt*s->atoms->p[iOff][2]*invMass;
                    }
                }
                stopTimer(positionTimer);
            }
        }
    }
}

/// Calculates total kinetic and potential energy across all tasks.  The
/// local potential energy is a by-product of the force routine.
void kineticEnergy(SimFlat* s)
{
    real3  *atomP = s->atoms->p;
    int *nAtoms = s->boxes->nAtoms;
    for(int z=0; z < s->boxes->gridSize[2]; z++) {
        for(int y=0; y < s->boxes->gridSize[1]; y++) {
            int rowBox = z*s->boxes->gridSize[1]*s->boxes->gridSize[0]+y*s->boxes->gridSize[0];
#pragma omp task depend(out: reductionArray[rowBox], reductionArrayInt[rowBox]) \
                 depend( in: atomP[rowBox*MAXATOMS], nAtoms[rowBox])
            {
                startTimer(KETimer);
                for(int iBox=rowBox; iBox < rowBox + s->boxes->gridSize[0]; iBox++) {
                    reductionArray[iBox] = 0.;
                    for (int iOff=MAXATOMS*iBox,ii=0; ii<s->boxes->nAtoms[iBox]; ii++,iOff++) {
                        int iSpecies = s->atoms->iSpecies[iOff];
                        real_t invMass = 0.5/s->species[iSpecies].mass;
                        reductionArray[iBox] += ( s->atoms->p[iOff][0] * s->atoms->p[iOff][0] +
                                                  s->atoms->p[iOff][1] * s->atoms->p[iOff][1] +
                                                  s->atoms->p[iOff][2] * s->atoms->p[iOff][2] )*invMass;
                    }
                    reductionArrayInt[iBox] = s->boxes->nAtoms[iBox];
                }
                stopTimer(KETimer);
            }
        }
    }
    ompReduceRowReal(reductionArray, s->boxes->gridSize);
    ompReduceRowInt(reductionArrayInt, s->boxes->gridSize);

    real_t *eKinetic= &(s->eKinetic);
    int *atomTotal = &(s->atoms->nLocal);
#pragma omp task depend( in: reductionArray[0] , reductionArrayInt[0]) \
                 depend( out: eKinetic[0] , atomTotal[0])
    {
        startTimer(KEReduceTimer);
        s->eKinetic = reductionArray[0];
        reductionArray[0] = 0;
        s->atoms->nLocal = reductionArrayInt[0];
        stopTimer(KEReduceTimer);
    }

}

/// \details
/// This function provides one-stop shopping for the sequence of events
/// that must occur for a proper exchange of halo atoms after the atom
/// positions have been updated by the integrator.
///
/// - updateLinkCells: Since atoms have moved, some may be in the wrong
///   link cells.
/// - haloExchange (atom version): Sends atom data to remote tasks. 
/// - sort: Sort the atoms.
///
/// \see updateLinkCells
/// \see initAtomHaloExchange
/// \see sortAtomsInCell
void redistributeAtoms(SimFlat* s)
{
    //TODO: re-organize this in a better way.
    updateLinkCells(s->boxes, s->boxesBuffer, s->atoms, s->atomsBuffer);
}
