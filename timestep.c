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
    //How do I combine these? 
    //TODO: Can I use a velocity + position task?
    //TODO: Can I use a force + velocity task?
    for (int ii=0; ii<nSteps; ++ii) {
        advanceVelocity(s, s->boxes->nLocalBoxes, 0.5*dt);//in: atomF, atomP, out: atomP
        advancePosition(s, s->boxes->nLocalBoxes, dt);    //in: atomP, out: atomR
        redistributeAtoms(s);                             //potentially entire atoms moved, but 27->1 deps
        computeForce(s);                                  //in: atomR, out: atomF, atomU, reduction, but 27->1 deps
        advanceVelocity(s, s->boxes->nLocalBoxes, 0.5*dt);//in: atomF, atomP, out: atomP

    }
    kineticEnergy(s);//reduction over atomP
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
    for (int iBox=0; iBox<nBoxes; iBox++) {
#pragma omp task depend(inout: atomP[iBox*MAXATOMS]) depend(in: atomF[iBox*MAXATOMS])
        {
            startTimer(velocityTimer);
            for (int iOff=MAXATOMS*iBox,ii=0; ii<s->boxes->nAtoms[iBox]; ii++,iOff++) {
                s->atoms->p[iOff][0] += dt*s->atoms->f[iOff][0];
                s->atoms->p[iOff][1] += dt*s->atoms->f[iOff][1];
                s->atoms->p[iOff][2] += dt*s->atoms->f[iOff][2];
            }
            stopTimer(velocityTimer);
        }
    }
}

void advancePosition(SimFlat* s, int nBoxes, real_t dt)
{
    real3 *atomP = s->atoms->p;
    real3 *atomR = s->atoms->r;
    for (int iBox=0; iBox<nBoxes; iBox++)
    {
#pragma omp task depend(inout: atomR[iBox*MAXATOMS]) depend(in: atomP[iBox*MAXATOMS])
        {
            startTimer(positionTimer);
            for (int iOff=MAXATOMS*iBox,ii=0; ii<s->boxes->nAtoms[iBox]; ii++,iOff++)
            {
                int iSpecies = s->atoms->iSpecies[iOff];
                real_t invMass = 1.0/s->species[iSpecies].mass;
                s->atoms->r[iOff][0] += dt*s->atoms->p[iOff][0]*invMass;
                s->atoms->r[iOff][1] += dt*s->atoms->p[iOff][1]*invMass;
                s->atoms->r[iOff][2] += dt*s->atoms->p[iOff][2]*invMass;
            }
            stopTimer(positionTimer);
        }
    }
}

/// Calculates total kinetic and potential energy across all tasks.  The
/// local potential energy is a by-product of the force routine.
void kineticEnergy(SimFlat* s)
{
    real3  *atomP = s->atoms->p;
    for (int iBox=0; iBox<s->boxes->nLocalBoxes; iBox++) {
#pragma omp task depend(out: reductionArray[iBox]) depend( in: atomP[iBox*MAXATOMS])
        {
            reductionArray[iBox] = 0.;
            for (int iOff=MAXATOMS*iBox,ii=0; ii<s->boxes->nAtoms[iBox]; ii++,iOff++) {
                int iSpecies = s->atoms->iSpecies[iOff];
                real_t invMass = 0.5/s->species[iSpecies].mass;
                reductionArray[iBox] += ( s->atoms->p[iOff][0] * s->atoms->p[iOff][0] +
                                          s->atoms->p[iOff][1] * s->atoms->p[iOff][1] +
                                          s->atoms->p[iOff][2] * s->atoms->p[iOff][2] )*invMass;
            }
        }
    }
    ompReduce(reductionArray, s->boxes->nLocalBoxes);
    real_t *eKinetic= &(s->eKinetic);
#pragma omp task depend( in: reductionArray[0] ) depend( out: eKinetic[0] )
    {
        s->eKinetic = reductionArray[0];
        reductionArray[0] = 0;
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
