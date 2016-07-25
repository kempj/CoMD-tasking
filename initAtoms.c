/// \file
/// Initialize the atom configuration.

#include "initAtoms.h"

#include <math.h>
#include <assert.h>

#include "constants.h"
#include "decomposition.h"
#include "parallel.h"
#include "random.h"
#include "linkCells.h"
#include "timestep.h"
#include "memUtils.h"
#include "performanceTimers.h"

//static void computeVcm(SimFlat* s, real_t vcm[3]);

extern struct SimFlatSt* sim;
extern real3 *r3ReductionArray;
double *reductionArray;

real_t vZero[3] = {0., 0., 0.};


/// \details
/// Call functions such as createFccLattice and setTemperature to set up
/// initial atom positions and momenta.
Atoms* initAtoms(LinkCell* boxes)
{
    Atoms* atoms = comdMalloc(sizeof(Atoms));

    int maxTotalAtoms = MAXATOMS*boxes->nTotalBoxes;

    atoms->gid =      (int*)   comdMalloc(maxTotalAtoms*sizeof(int));
    atoms->iSpecies = (int*)   comdMalloc(maxTotalAtoms*sizeof(int));
    atoms->r =        (real3*) comdMalloc(maxTotalAtoms*sizeof(real3));
    atoms->p =        (real3*) comdMalloc(maxTotalAtoms*sizeof(real3));
    atoms->f =        (real3*) comdMalloc(maxTotalAtoms*sizeof(real3));
    atoms->U =        (real_t*)comdMalloc(maxTotalAtoms*sizeof(real_t));

    atoms->nLocal = 0;
    atoms->nGlobal = 0;

    for (int iOff = 0; iOff < maxTotalAtoms; iOff++)
    {
        atoms->gid[iOff] = 0;
        atoms->iSpecies[iOff] = 0;
        zeroReal3(atoms->r[iOff]);
        zeroReal3(atoms->p[iOff]);
        zeroReal3(atoms->f[iOff]);
        atoms->U[iOff] = 0.;
    }

    return atoms;
}

void destroyAtoms(Atoms *atoms)
{
    freeMe(atoms,gid);
    freeMe(atoms,iSpecies);
    freeMe(atoms,r);
    freeMe(atoms,p);
    freeMe(atoms,f);
    freeMe(atoms,U);
    comdFree(atoms);
}

/// Creates atom positions on a face centered cubic (FCC) lattice with
/// nx * ny * nz unit cells and lattice constant lat.
/// Set momenta to zero.
void createFccLattice(int nx, int ny, int nz, real_t lat, SimFlat* s)
{
    const real_t* localMin = s->domain->localMin; // alias
    const real_t* localMax = s->domain->localMax; // alias

    int nb = 4; // number of atoms in the basis
    real3 basis[4] = { {0.25, 0.25, 0.25},
        {0.25, 0.75, 0.75},
        {0.75, 0.25, 0.75},
        {0.75, 0.75, 0.25} };

    // create and place atoms
    int begin[3];
    int end[3];
    for (int ii=0; ii<3; ++ii)
    {
        begin[ii] = floor(localMin[ii]/lat);
        end[ii]   = ceil (localMax[ii]/lat);
    }

    real_t px,py,pz;
    px=py=pz=0.0;
    for (int ix=begin[0]; ix<end[0]; ++ix)
        for (int iy=begin[1]; iy<end[1]; ++iy)
            for (int iz=begin[2]; iz<end[2]; ++iz)
                for (int ib=0; ib<nb; ++ib)
                {
                    real_t rx = (ix+basis[ib][0]) * lat;
                    real_t ry = (iy+basis[ib][1]) * lat;
                    real_t rz = (iz+basis[ib][2]) * lat;
                    if (rx < localMin[0] || rx >= localMax[0]) continue;
                    if (ry < localMin[1] || ry >= localMax[1]) continue;
                    if (rz < localMin[2] || rz >= localMax[2]) continue;
                    int id = ib+nb*(iz+nz*(iy+ny*(ix)));
                    putAtomInBox(sim->boxes, sim->atoms, id, 0, rx, ry, rz, px, py, pz);
                }


    // set total atoms in simulation
    startTimer(commReduceTimer);
    addIntParallel(&sim->atoms->nLocal, &sim->atoms->nGlobal, 1);
    stopTimer(commReduceTimer);

    assert(sim->atoms->nGlobal == nb*nx*ny*nz);
}

/// Sets the center of mass velocity of the system.
/// \param [in] newVcm The desired center of mass velocity.
void setVcm()
{
#pragma omp task depend(inout: reductionArray[0], r3ReductionArray[0][0])
    {
        reductionArray[0] = 0.;
        r3ReductionArray[0][0] = 0.;
        r3ReductionArray[0][1] = 0.;
        r3ReductionArray[0][2] = 0.;
    }
    real3 *atomP = sim->atoms->p;
    for (int iBox=0; iBox < sim->boxes->nLocalBoxes; ++iBox) {
#pragma omp task depend( in: atomP[iBox]) depend( out: r3ReductionArray[iBox], reductionArray[iBox] )
        {
            int Off = MAXATOMS*iBox;
            for (int ii=0; ii < sim->boxes->nAtoms[iBox]; ++ii) {
                r3ReductionArray[iBox][0] += sim->atoms->p[Off+ii][0];
                r3ReductionArray[iBox][1] += sim->atoms->p[Off+ii][1];
                r3ReductionArray[iBox][2] += sim->atoms->p[Off+ii][2];

                int iSpecies = sim->atoms->iSpecies[Off+ii];
                reductionArray[iBox] += sim->species[iSpecies].mass;
            }
            //printf("finishing Vcm loop 1; in : atomP[%d](%p), out r3[%d](%p), r[%d](%p)\n",iBox, &atomP[iBox], 
//                                                                                           iBox, &r3ReductionArray[iBox], 
//                                                                                           iBox, &reductionArray[iBox]);
        }
    }
    ompReduceStride(r3ReductionArray[0], sim->boxes->nLocalBoxes, 3);
    ompReduce(reductionArray, sim->boxes->nLocalBoxes);//NOTE: might want to combine these.

#pragma omp task depend( in: r3ReductionArray[0], reductionArray[0]) depend( out: vZero[0])
    {
        real_t v3 = reductionArray[0]; 
        vZero[0] -= r3ReductionArray[0][0]/v3;
        vZero[1] -= r3ReductionArray[0][1]/v3;
        vZero[2] -= r3ReductionArray[0][2]/v3;
    }

    for (int iBox=0; iBox<sim->boxes->nLocalBoxes; ++iBox) {
#pragma omp task depend(inout: atomP[iBox]) depend( in: vZero[0])
        {
        for (int iOff=MAXATOMS*iBox, ii=0; ii<sim->boxes->nAtoms[iBox]; ++ii, ++iOff) {
            int iSpecies = sim->atoms->iSpecies[iOff];
            real_t mass = sim->species[iSpecies].mass;

            sim->atoms->p[iOff][0] += mass * vZero[0];
            sim->atoms->p[iOff][1] += mass * vZero[1];
            sim->atoms->p[iOff][2] += mass * vZero[2];
        }
            //printf("finishing Vcm loop 2; in : atomP[%d](%p)\n", iBox, &atomP[iBox]);
        }
    }
}

void setTemperature(real_t temperature)
{
    // set initial velocities for the distribution
    real3 *atomP = sim->atoms->p;
    for (int iBox=0; iBox<sim->boxes->nLocalBoxes; ++iBox) {
#pragma omp task firstprivate(iBox) depend(out: atomP[iBox])
        for (int iOff=MAXATOMS*iBox, ii=0; ii<sim->boxes->nAtoms[iBox]; ++ii, ++iOff) {
            int iType = sim->atoms->iSpecies[iOff];
            real_t mass = sim->species[iType].mass;
            real_t sigma = sqrt(kB_eV * temperature/mass);
            uint64_t seed = mkSeed(sim->atoms->gid[iOff], 123);
            sim->atoms->p[iOff][0] = mass * sigma * gasdev(&seed);
            sim->atoms->p[iOff][1] = mass * sigma * gasdev(&seed);
            sim->atoms->p[iOff][2] = mass * sigma * gasdev(&seed);
        }
    }
    if (temperature == 0.0)
        return;
    setVcm();//atomP inout -> reduction -> atomP inout
    kineticEnergy(sim);//atomP reduced into ePotential
    
    real_t temp = (sim->eKinetic/sim->atoms->nGlobal)/kB_eV/1.5;
    real_t scaleFactor = sqrt(temperature/temp);
    for (int iBox=0; iBox<sim->boxes->nLocalBoxes; ++iBox) {
#pragma omp task depend(inout: atomP[iBox])
        for (int iOff=MAXATOMS*iBox, ii=0; ii<sim->boxes->nAtoms[iBox]; ++ii, ++iOff) {
            sim->atoms->p[iOff][0] *= scaleFactor;
            sim->atoms->p[iOff][1] *= scaleFactor;
            sim->atoms->p[iOff][2] *= scaleFactor;
        }
    }
    kineticEnergy(sim);
    real_t *eKinetic= &(sim->eKinetic);
#pragma omp task depend(in: eKinetic)
    temp = sim->eKinetic/sim->atoms->nGlobal/kB_eV/1.5;
    //printf("eKinetic = %f, ePotential = %f\n",sim->eKinetic, sim->ePotential);
}

/// Add a random displacement to the atom positions.
/// Atoms are displaced by a random distance in the range
/// [-delta, +delta] along each axis.
/// \param [in] delta The maximum displacement (along each axis).
void randomDisplacements(real_t delta)
{
    real3 *atomR = sim->atoms->r;
    for (int iBox=0; iBox<sim->boxes->nLocalBoxes; ++iBox) {
#pragma omp task depend(inout: atomR[iBox][0])
        for (int iOff=MAXATOMS*iBox, ii=0; ii<sim->boxes->nAtoms[iBox]; ++ii, ++iOff) {
            uint64_t seed = mkSeed(sim->atoms->gid[iOff], 457);
            sim->atoms->r[iOff][0] += (2.0*lcg61(&seed)-1.0) * delta;
            sim->atoms->r[iOff][1] += (2.0*lcg61(&seed)-1.0) * delta;
            sim->atoms->r[iOff][2] += (2.0*lcg61(&seed)-1.0) * delta;
        }
    }
}

