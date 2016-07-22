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
                    putAtomInBox(s->boxes, s->atoms, id, 0, rx, ry, rz, px, py, pz);
                }

    // set total atoms in simulation
    startTimer(commReduceTimer);
    addIntParallel(&s->atoms->nLocal, &s->atoms->nGlobal, 1);
    stopTimer(commReduceTimer);

    assert(s->atoms->nGlobal == nb*nx*ny*nz);
}

/// Sets the center of mass velocity of the system.
/// \param [in] newVcm The desired center of mass velocity.
void setVcm(struct SimFlatSt* s, real_t vcm[3])
{
    real3 *atomP = s->atoms->p;
    for (int iBox=0; iBox<s->boxes->nLocalBoxes; ++iBox) {
#pragma omp task depend( in: atomP[iBox]) depend( out: r3ReductionArray[iBox], reductionArray[iBox] )
        for (int iOff=MAXATOMS*iBox, ii=0; ii<s->boxes->nAtoms[iBox]; ++ii, ++iOff) {
            r3ReductionArray[iBox][0] += s->atoms->p[iOff][0];
            r3ReductionArray[iBox][1] += s->atoms->p[iOff][1];
            r3ReductionArray[iBox][2] += s->atoms->p[iOff][2];

            int iSpecies = s->atoms->iSpecies[iOff];
            reductionArray[iOff] += s->species[iSpecies].mass;
        }
    }
    ompReduceStride(r3ReductionArray[0], s->boxes->nLocalBoxes, 3);
    ompReduce(reductionArray, s->boxes->nLocalBoxes);//NOTE: might want to combine these.

#pragma omp task depend( in: r3ReductionArray[0], reductionArray[0]) depend( out: vcm[0])
    {
        real_t v3 = reductionArray[0]; 
        vcm[0] -= r3ReductionArray[0][0]/v3;
        vcm[1] -= r3ReductionArray[0][1]/v3;
        vcm[2] -= r3ReductionArray[0][2]/v3;
    }

    for (int iBox=0; iBox<s->boxes->nLocalBoxes; ++iBox) {
#pragma omp task depend(inout: atomP[iBox][0]) depend( in: vcm[0])
        for (int iOff=MAXATOMS*iBox, ii=0; ii<s->boxes->nAtoms[iBox]; ++ii, ++iOff) {
            int iSpecies = s->atoms->iSpecies[iOff];
            real_t mass = s->species[iSpecies].mass;

            s->atoms->p[iOff][0] += mass * vcm[0];
            s->atoms->p[iOff][1] += mass * vcm[1];
            s->atoms->p[iOff][2] += mass * vcm[2];
        }
    }
}

/// Sets the temperature of system.
///
/// Selects atom velocities randomly from a boltzmann (equilibrium)
/// distribution that corresponds to the specified temperature.  This
/// random process will typically result in a small, but non zero center
/// of mass velocity and a small difference from the specified
/// temperature.  For typical MD runs these small differences are
/// unimportant, However, to avoid possible confusion, we set the center
/// of mass velocity to zero and scale the velocities to exactly match
/// the input temperature.
void setTemperature(SimFlat* s, real_t temperature)
{
    // set initial velocities for the distribution
    real3 *atomP = s->atoms->p;
    for (int iBox=0; iBox<s->boxes->nLocalBoxes; ++iBox) {
#pragma omp task depend(out: atomP[iBox][0])
        for (int iOff=MAXATOMS*iBox, ii=0; ii<s->boxes->nAtoms[iBox]; ++ii, ++iOff) {
            int iType = s->atoms->iSpecies[iOff];
            real_t mass = s->species[iType].mass;
            real_t sigma = sqrt(kB_eV * temperature/mass);
            uint64_t seed = mkSeed(s->atoms->gid[iOff], 123);
            s->atoms->p[iOff][0] = mass * sigma * gasdev(&seed);
            s->atoms->p[iOff][1] = mass * sigma * gasdev(&seed);
            s->atoms->p[iOff][2] = mass * sigma * gasdev(&seed);
        }
    }
    if (temperature == 0.0)
        return;
    setVcm(s, &(vZero[0]));//atomP inout
    kineticEnergy(s);//parallel for reduce(serialized)
    
    real_t temp = (s->eKinetic/s->atoms->nGlobal)/kB_eV/1.5;
    real_t scaleFactor = sqrt(temperature/temp);
    for (int iBox=0; iBox<s->boxes->nLocalBoxes; ++iBox) {
#pragma omp task depend(inout: atomP[iBox][0])
        for (int iOff=MAXATOMS*iBox, ii=0; ii<s->boxes->nAtoms[iBox]; ++ii, ++iOff) {
            s->atoms->p[iOff][0] *= scaleFactor;
            s->atoms->p[iOff][1] *= scaleFactor;
            s->atoms->p[iOff][2] *= scaleFactor;
        }
    }
    kineticEnergy(s);
    temp = s->eKinetic/s->atoms->nGlobal/kB_eV/1.5;
}

/// Add a random displacement to the atom positions.
/// Atoms are displaced by a random distance in the range
/// [-delta, +delta] along each axis.
/// \param [in] delta The maximum displacement (along each axis).
void randomDisplacements(SimFlat* s, real_t delta)
{
//#pragma omp parallel for
    real3 *atomR = s->atoms->r;
    for (int iBox=0; iBox<s->boxes->nLocalBoxes; ++iBox) {
#pragma omp task depend(inout: atomR[iBox][0])
        for (int iOff=MAXATOMS*iBox, ii=0; ii<s->boxes->nAtoms[iBox]; ++ii, ++iOff) {
            uint64_t seed = mkSeed(s->atoms->gid[iOff], 457);
            s->atoms->r[iOff][0] += (2.0*lcg61(&seed)-1.0) * delta;
            s->atoms->r[iOff][1] += (2.0*lcg61(&seed)-1.0) * delta;
            s->atoms->r[iOff][2] += (2.0*lcg61(&seed)-1.0) * delta;
        }
    }
}

