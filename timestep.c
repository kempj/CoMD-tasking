/// \file
/// Leapfrog time integrator

#include "timestep.h"

#include <omp.h>

#include "CoMDTypes.h"
#include "linkCells.h"
#include "parallel.h"
#include "performanceTimers.h"

static void advanceVelPos(SimFlat* s, real_t dtVel, real_t dtPos);
static void advanceVelocity(SimFlat* s, real_t dt);
static void advancePosition(SimFlat* s, real_t dt);

extern double *reductionArray;
extern int *reductionArrayInt;
extern double globalEnergy;

/// Advance the simulation time to t+dt using a leap frog method
///
/// This was reworked for the tasking version to replace a chain of 5 tasks with 3 tasks.
/// This should improve locality and decrease task overhead.
double timestep(SimFlat* s, int nSteps, real_t dt)
{
    advanceVelPos(s, 0.5*dt, dt);//in: atomF, atomP, out: atomP atomR
    for (int ii=0; ii<nSteps-1; ++ii) {
        redistributeAtoms(s);                             //potentially entire atoms moved, but 9->1 deps
        computeForce(s);                                  //in: atomR, out: atomF, atomU, reduction, but 9->1 deps
        advanceVelPos(s, dt, dt);//in: atomF, atomP, out: atomP atomR
    }
    redistributeAtoms(s);                             //potentially entire atoms moved, but 9->1 deps
    computeForce(s);                                  //in: atomR, out: atomF, atomU, reduction, but 9->1 deps
    advanceVelocity(s, 0.5*dt);//in: atomF, atomP, out: atomP
    kineticEnergy(s);//atomP -> KE and nAtoms -> nLocal
    return s->ePotential;
}

void computeForce(SimFlat* s)
{
    s->pot->force(s);
}

void cellVelocity(Atoms *atoms, int nAtoms, int iBox, real_t dt) {
    for (int iOff=MAXATOMS*iBox,ii=0; ii<nAtoms; ii++,iOff++) {
        atoms->p[iOff][0] += dt*atoms->f[iOff][0];
        atoms->p[iOff][1] += dt*atoms->f[iOff][1];
        atoms->p[iOff][2] += dt*atoms->f[iOff][2];
    }
}

void cellPosition(Atoms *atoms, SpeciesData * species, int nAtoms, int iBox, real_t dt) 
{
    for (int iOff=MAXATOMS*iBox,ii=0; ii<nAtoms; ii++,iOff++) {
        int iSpecies = atoms->iSpecies[iOff];
        real_t invMass = 1.0/species[iSpecies].mass;
        atoms->r[iOff][0] += dt*atoms->p[iOff][0]*invMass;
        atoms->r[iOff][1] += dt*atoms->p[iOff][1]*invMass;
        atoms->r[iOff][2] += dt*atoms->p[iOff][2]*invMass;
    }
}

void advanceVelPos(SimFlat* s, real_t dtVel, real_t dtPos)
{
    real3 *atomP = s->atoms->p;
    real3 *atomR = s->atoms->r;
    real3 *atomF = s->atoms->f;
    for(int z=0; z < s->boxes->gridSize[2]; z++) {
        for(int y=0; y < s->boxes->gridSize[1]; y++) {
            int rowBox = z*s->boxes->gridSize[1]*s->boxes->gridSize[0] + y*s->boxes->gridSize[0];
#pragma omp task depend(inout: atomP[rowBox*MAXATOMS], atomR[rowBox*MAXATOMS]) \
                 depend(in: atomF[rowBox*MAXATOMS])
            {
                startTimer(velPosTimer);
                for(int iBox=rowBox; iBox < rowBox + s->boxes->gridSize[0]; iBox++) {
                    cellVelocity(s->atoms, s->boxes->nAtoms[iBox], iBox, dtVel);
                    //for (int iOff=MAXATOMS*iBox,ii=0; ii<s->boxes->nAtoms[iBox]; ii++,iOff++) {
                    //    s->atoms->p[iOff][0] += dtVel*s->atoms->f[iOff][0];
                    //    s->atoms->p[iOff][1] += dtVel*s->atoms->f[iOff][1];
                    //    s->atoms->p[iOff][2] += dtVel*s->atoms->f[iOff][2];
                    //}

                    cellPosition(s->atoms, s->species, s->boxes->nAtoms[iBox], iBox, dtPos);
                    //for (int iOff=MAXATOMS*iBox,ii=0; ii<s->boxes->nAtoms[iBox]; ii++,iOff++) {
                    //    int iSpecies = s->atoms->iSpecies[iOff];
                    //    real_t invMass = 1.0/s->species[iSpecies].mass;
                    //    s->atoms->r[iOff][0] += dtPos*s->atoms->p[iOff][0]*invMass;
                    //    s->atoms->r[iOff][1] += dtPos*s->atoms->p[iOff][1]*invMass;
                    //    s->atoms->r[iOff][2] += dtPos*s->atoms->p[iOff][2]*invMass;
                    //}
                }
                stopTimer(velPosTimer);
            }
        }
    }
}

void advanceVelocity(SimFlat* s, real_t dt)
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
                    cellVelocity(s->atoms, s->boxes->nAtoms[iBox], iBox, dt);
                    //for (int iOff=MAXATOMS*iBox,ii=0; ii<s->boxes->nAtoms[iBox]; ii++,iOff++) {
                    //    s->atoms->p[iOff][0] += dt*s->atoms->f[iOff][0];
                    //    s->atoms->p[iOff][1] += dt*s->atoms->f[iOff][1];
                    //    s->atoms->p[iOff][2] += dt*s->atoms->f[iOff][2];
                    //}
                }
                stopTimer(velocityTimer);
            }
        }
    }
}

void advancePosition(SimFlat* s, real_t dt)
{
    real3 *atomP = s->atoms->p;
    real3 *atomR = s->atoms->r;
    for(int z=0; z < s->boxes->gridSize[2]; z++) {
        for(int y=0; y < s->boxes->gridSize[1]; y++) {
            int rowBox = z*s->boxes->gridSize[1]*s->boxes->gridSize[0]+y*s->boxes->gridSize[0];
#pragma omp task depend(inout: atomR[rowBox*MAXATOMS]) depend(in: atomP[rowBox*MAXATOMS])
            {
                startTimer(positionTimer);
                for(int iBox=rowBox; iBox < rowBox + s->boxes->gridSize[0]; iBox++) {
                    cellPosition(s->atoms, s->species, s->boxes->nAtoms[iBox], iBox, dt);
                    //for (int iOff=MAXATOMS*iBox,ii=0; ii<s->boxes->nAtoms[iBox]; ii++,iOff++) {
                    //    int iSpecies = s->atoms->iSpecies[iOff];
                    //    real_t invMass = 1.0/s->species[iSpecies].mass;
                    //    s->atoms->r[iOff][0] += dt*s->atoms->p[iOff][0]*invMass;
                    //    s->atoms->r[iOff][1] += dt*s->atoms->p[iOff][1]*invMass;
                    //    s->atoms->r[iOff][2] += dt*s->atoms->p[iOff][2]*invMass;
                    //}
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
///   link cells. This also sorts now.
///
/// \see updateLinkCells
void redistributeAtoms(SimFlat* s)
{
    //TODO: re-organize this in a better way.
    updateLinkCells(s->boxes, s->boxesBuffer, s->atoms, s->atomsBuffer);
}
