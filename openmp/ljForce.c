/// \file
/// Computes forces for the 12-6 Lennard Jones (LJ) potential.
///
/// The Lennard-Jones model is not a good representation for the
/// bonding in copper, its use has been limited to constant volume
/// simulations where the embedding energy contribution to the cohesive
/// energy is not included in the two-body potential
///
/// The parameters here are taken from Wolf and Phillpot and fit to the
/// room temperature lattice constant and the bulk melt temperature
/// Ref: D. Wolf and S.Yip eds. Materials Interfaces (Chapman & Hall
///      1992) Page 230.
///
/// Notes on LJ:
///
/// http://en.wikipedia.org/wiki/Lennard_Jones_potential
///
/// The total inter-atomic potential energy in the LJ model is:
///
/// \f[
///   E_{tot} = \sum_{ij} U_{LJ}(r_{ij})
/// \f]
/// \f[
///   U_{LJ}(r_{ij}) = 4 \epsilon
///           \left\{ \left(\frac{\sigma}{r_{ij}}\right)^{12}
///           - \left(\frac{\sigma}{r_{ij}}\right)^6 \right\}
/// \f]
///
/// where \f$\epsilon\f$ and \f$\sigma\f$ are the material parameters in the potential.
///    - \f$\epsilon\f$ = well depth
///    - \f$\sigma\f$   = hard sphere diameter
///
///  To limit the interation range, the LJ potential is typically
///  truncated to zero at some cutoff distance. A common choice for the
///  cutoff distance is 2.5 * \f$\sigma\f$.
///  This implementation can optionally shift the potential slightly
///  upward so the value of the potential is zero at the cuotff
///  distance.  This shift has no effect on the particle dynamics.
///
///
/// The force on atom i is given by
///
/// \f[
///   F_i = -\nabla_i \sum_{jk} U_{LJ}(r_{jk})
/// \f]
///
/// where the subsrcipt i on the gradient operator indicates that the
/// derivatives are taken with respect to the coordinates of atom i.
/// Liberal use of the chain rule leads to the expression
///
/// \f{eqnarray*}{
///   F_i &=& - \sum_j U'_{LJ}(r_{ij})\hat{r}_{ij}\\
///       &=& \sum_j 24 \frac{\epsilon}{r_{ij}} \left\{ 2 \left(\frac{\sigma}{r_{ij}}\right)^{12}
///               - \left(\frac{\sigma}{r_{ij}}\right)^6 \right\} \hat{r}_{ij}
/// \f}
///
/// where \f$\hat{r}_{ij}\f$ is a unit vector in the direction from atom
/// i to atom j.
/// 
///

#include "ljForce.h"
#include "performanceTimers.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <omp.h>

#include "constants.h"
#include "mytype.h"
#include "parallel.h"
#include "linkCells.h"
#include "memUtils.h"
#include "CoMDTypes.h"

#define POT_SHIFT 1.0

/// Derived struct for a Lennard Jones potential.
/// Polymorphic with BasePotential.
/// \see BasePotential
typedef struct LjPotentialSt
{
    real_t cutoff;          //!< potential cutoff distance in Angstroms
    real_t mass;            //!< mass of atoms in intenal units
    real_t lat;             //!< lattice spacing (angs) of unit cell
    char latticeType[8];    //!< lattice type, e.g. FCC, BCC, etc.
    char  name[3];	   //!< element name
    int	 atomicNo;	   //!< atomic number  
    int  (*force)(SimFlat* s); //!< function pointer to force routine
    void (*print)(FILE* file, BasePotential* pot);
    void (*destroy)(BasePotential** pot); //!< destruction of the potential
    real_t sigma;
    real_t epsilon;
} LjPotential;

static int ljForce(SimFlat* s);
static void ljPrint(FILE* file, BasePotential* pot);


extern double *reductionArray;

real_t rCut;
real_t sigma;
real_t epsilon;
real_t s6;
real_t rCut2;
real_t rCut6;
real_t eShift;

void ljDestroy(BasePotential** inppot)
{
    if ( ! inppot ) return;
    LjPotential* pot = (LjPotential*)(*inppot);
    if ( ! pot ) return;
    comdFree(pot);
    *inppot = NULL;

    return;
}

/// Initialize an Lennard Jones potential for Copper.
BasePotential* initLjPot(void)
{
    LjPotential *pot = (LjPotential*)comdMalloc(sizeof(LjPotential));
    pot->force = ljForce;
    pot->print = ljPrint;
    pot->destroy = ljDestroy;
    pot->sigma = 2.315;	                  // Angstrom
    pot->epsilon = 0.167;                  // eV
    pot->mass = 63.55 * amuToInternalMass; // Atomic Mass Units (amu)

    pot->lat = 3.615;                      // Equilibrium lattice const in Angs
    strcpy(pot->latticeType, "FCC");       // lattice type, i.e. FCC, BCC, etc.
    pot->cutoff = 2.5*pot->sigma;          // Potential cutoff in Angs

    strcpy(pot->name, "Cu");
    pot->atomicNo = 29;
    
    rCut = pot->cutoff;
    sigma = pot->sigma;
    epsilon = pot->epsilon;
    s6 = sigma*sigma*sigma*sigma*sigma*sigma;
    rCut2 = rCut*rCut;
    rCut6 = s6 / (rCut2*rCut2*rCut2);
    eShift = POT_SHIFT * rCut6 * (rCut6 - 1.0);

    return (BasePotential*) pot;
}

void ljPrint(FILE* file, BasePotential* pot)
{
    LjPotential* ljPot = (LjPotential*) pot;
    fprintf(file, "  Potential type   : Lennard-Jones\n");
    fprintf(file, "  Species name     : %s\n", ljPot->name);
    fprintf(file, "  Atomic number    : %d\n", ljPot->atomicNo);
    fprintf(file, "  Mass             : "FMT1" amu\n", ljPot->mass / amuToInternalMass); // print in amu
    fprintf(file, "  Lattice Type     : %s\n", ljPot->latticeType);
    fprintf(file, "  Lattice spacing  : "FMT1" Angstroms\n", ljPot->lat);
    fprintf(file, "  Cutoff           : "FMT1" Angstroms\n", ljPot->cutoff);
    fprintf(file, "  Epsilon          : "FMT1" eV\n", ljPot->epsilon);
    fprintf(file, "  Sigma            : "FMT1" Angstroms\n", ljPot->sigma);
}


//calculates the force on atoms in a box.
real_t boxForce(int iBox, SimFlat *s)
{
    const int* gridSize = s->boxes->gridSize;
    const real_t* localMax = s->boxes->localMax;

    int xyz[3];
    xyz[0] = iBox % gridSize[0];
    int tmpBox = iBox / gridSize[0];
    xyz[1] = tmpBox % gridSize[1];
    xyz[2] = tmpBox / gridSize[1];

    //Offset is needed because halo cells are not used.
    //Instead of copying the local cell corresponding to a halo neighbor into the halo, 
    //  the logic is changed to look at the local cell, and offset the position of the atoms in
    //  place, without altering the local cell being read from.
    real3 offset[3][3][3];

    int ijk[3];
    for(int i=0; i<3; i++) {
        ijk[0] = i;
        for(int j=0; j<3; j++) {
            ijk[1] = j;
            for(int k=0; k<3; k++) {
                ijk[2] = k;
                for(int m=0; m<3; m++) {
                    if(ijk[m]+xyz[m]-1 == gridSize[m]) {
                        offset[i][j][k][m] = localMax[m];
                    } else if(ijk[m]+xyz[m]-1 == -1) {
                        offset[i][j][k][m] =-localMax[m];
                    } else {
                        offset[i][j][k][m] = 0;
                    }
                }
            }
        }
    }

    int nIBox = s->boxes->nAtoms[iBox];
    real_t ePot = 0;
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            for(int k=0; k<3; k++) {
                int realJBox = getBoxFromTuple(s->boxes, i+xyz[0]-1, j+xyz[1]-1, k+xyz[2]-1);
                int jBox = getLocalHaloTuple(s->boxes, realJBox);
                int nJBox = s->boxes->nAtoms[jBox];
                for(int iOff=MAXATOMS*iBox; iOff<(iBox*MAXATOMS+nIBox); iOff++) {
                    for(int jOff=jBox*MAXATOMS; jOff<(jBox*MAXATOMS+nJBox); jOff++) {
                        real3 dr;
                        real_t r2 = 0.0;
                        for(int m=0; m<3; m++) {
                            dr[m] = s->atoms->r[iOff][m] - (s->atoms->r[jOff][m] + offset[i][j][k][m]);
                            r2+=dr[m]*dr[m];
                        }
                        if(r2<=rCut2 && r2>0.0) {
                            r2 = 1.0/r2;
                            real_t r6 = s6 * (r2*r2*r2);
                            real_t eLocal = r6 * (r6 - 1.0) - eShift;
                            ePot += 0.5*eLocal;

                            real_t fr = - 4.0*epsilon*r6*r2*(12.0*r6 - 6.0);
                            for (int m=0; m<3; m++) {
                                s->atoms->f[iOff][m] -= dr[m]*fr;
                            }
                        }
                    }
                }
            }
        }
    }
    return ePot;
}

int ljForce(SimFlat* s)
{
    int *gridSize = s->boxes->gridSize;
    real3  *atomF = s->atoms->f;
    real3  *atomR = s->atoms->r;
    int dep[9];

    for(int z=0; z < s->boxes->gridSize[2]; z++) {
        for(int y=0; y < s->boxes->gridSize[1]; y++) {
            int rowBox = z*s->boxes->gridSize[1]*s->boxes->gridSize[0] + y*s->boxes->gridSize[0];
            getNeighborRows(s->boxes, y, z, dep);
#pragma omp task depend(out: atomF[rowBox*MAXATOMS], reductionArray[rowBox]) \
                 depend( in: atomR[dep[0]*MAXATOMS], atomR[dep[1]*MAXATOMS], atomR[dep[2]*MAXATOMS], \
                             atomR[dep[3]*MAXATOMS], atomR[dep[4]*MAXATOMS], atomR[dep[5]*MAXATOMS], \
                             atomR[dep[6]*MAXATOMS], atomR[dep[7]*MAXATOMS], atomR[dep[8]*MAXATOMS] )
            {
                startTimer(computeForceTimer);
                real_t ePot = 0;
                for(int iBox=rowBox; iBox < rowBox + s->boxes->gridSize[0]; iBox++) {
                    //TODO: remove this once tested with new copyatom
                    for(int ii=iBox*MAXATOMS; ii<(iBox+1)*MAXATOMS;ii++) {
                        zeroReal3(s->atoms->f[ii]);
                    }
                    ePot += boxForce(iBox, s);
                }
                reductionArray[rowBox] = ePot;
                stopTimer(computeForceTimer);
            }
        }
    }
    ompReduceReal(reductionArray, s->boxes->nLocalBoxes, gridSize[0]);

    real_t *ePotential = &(s->ePotential);
#pragma omp task depend(inout: reductionArray[0]) depend(out: ePotential[0])
    {
        startTimer(ePotReductionTimer);
        *ePotential = reductionArray[0]*4.0*((LjPotential*)(s->pot))->epsilon;
        reductionArray[0] = 0;
        stopTimer(ePotReductionTimer);
    }

    return 0;
}

real_t boxForcePart(SimFlat *s, int iBox, real3 iOffset, int jBox, real3 jOffset)
{
    int nIBox = s->boxes->nAtoms[iBox];
    real_t ePot = 0;

    int nJBox = s->boxes->nAtoms[jBox];
    for(int iOff=MAXATOMS*iBox; iOff<(iBox*MAXATOMS+nIBox); iOff++) {
        for(int jOff=jBox*MAXATOMS; jOff<(jBox*MAXATOMS+nJBox); jOff++) {
            real3 dr;
            real_t r2 = 0.0;
            for(int m=0; m<3; m++) {
                dr[m] = (s->atoms->r[iOff][m] + iOffset[m])- (s->atoms->r[jOff][m] + jOffset[m]);
                r2+=dr[m]*dr[m];
            }
            if(r2<=rCut2 && r2>0.0) {
                r2 = 1.0/r2;
                real_t r6 = s6 * (r2*r2*r2);
                real_t eLocal = r6 * (r6 - 1.0) - eShift;
                //Changed to account for any pair of cells only interacting once.
                //ePot += 0.5*eLocal;
                ePot += eLocal;

                real_t fr = - 4.0*epsilon*r6*r2*(12.0*r6 - 6.0);
                for (int m=0; m<3; m++) {
                    s->atoms->f[iOff][m] -= dr[m]*fr;
                    s->atoms->f[jOff][m] -= dr[m]*fr;
                }
            }
        }
    }
    return ePot;
}

//This might be better to send iBox and then an int[4] offset
// as long as we shift correctly we can assume dep[0] is local.
//void clusterForce(SimFlat *s, int dep[4], real_t offsetY, real_t offsetZ)
void clusterForce(SimFlat *s, int y, int z)
{
    real3  *atomF = s->atoms->f;
    real3  *atomR = s->atoms->r;

    int *gridSize = s->boxes->gridSize;
    int sizeZ = gridSize[1]*gridSize[0];
    int sizeY = gridSize[0];

    int dep[4];
    dep[0] = z*sizeZ + y*sizeY;
    dep[1] = dep[0] + sizeY;        
    dep[2] = dep[0] + sizeZ;        
    dep[3] = dep[0] + sizeY + sizeZ;


    if(y+1 == gridSize[1]) {
        dep[1] = dep[0] - (y*sizeY);
        dep[3] = dep[1] + sizeZ;
    }
    if(z+1 == gridSize[2]) {
        dep[2] -= z*sizeZ;
        dep[3] -= z*sizeZ;
    }


#pragma omp task depend(inout: reductionArray[dep[0]]) \
                 depend(  out: atomF[dep[0]*MAXATOMS], atomF[dep[1]*MAXATOMS], \
                               atomF[dep[2]*MAXATOMS], atomF[dep[3]*MAXATOMS]) \
                 depend(   in: atomR[dep[0]*MAXATOMS], atomR[dep[1]*MAXATOMS], \
                               atomR[dep[2]*MAXATOMS], atomR[dep[3]*MAXATOMS])
    {
        startTimer(computeForceTimer);

        real_t ePot = 0;
        real3 offset[4];
        for(int i=0; i < 4; i++)
            zeroReal3(offset[i]);
        offset[1][1] = offsetY;
        offset[2][2] = offsetZ;
        offset[3][1] = offsetY;
        offset[3][2] = offsetZ;

        int offsetX = s->boxes->localMax[0];
        int sizeX = s->boxes->gridSize[0];

        for(int i=0; i<sizeX-1; i++) {
            //row i with row i
            for(int j=1; j<4; j++) {
                ePot += boxForcePart(s, dep[0]+i, offset[0], dep[j] + i, offset[j]);
            }
            //row i with row i+1
            for(int j=0; j<4; j++) {
                for(int k=0; k<4; k++) {
                    ePot += boxForcePart(s, dep[j]+i, offset[j], dep[k]+i, offset[k]);
                }
            }
        }

        //last row with last row
        for(int j=1; j<4; j++) {
            ePot += boxForcePart(s, dep[0]+sizeX-1, offset[0], dep[j] + sizeX-1, offset[j]);
        }
        real3 tmpOffset = {offsetX,0,0};
        //last row with first row
        for(int j=0; j<4; j++) {
            for(int k=0; k<4; k++) {
                tmpOffset[1] = offset[k][1];
                tmpOffset[2] = offset[k][2];
                ePot += boxForcePart(s, dep[j]+sizeX-1, offset[j], dep[k], tmpOffset);
            }
        }
        reductionArray[dep[0]] += ePot;

        stopTimer(computeForceTimer);
    }
}

int ljForcePartial(SimFlat *s)
{
    int *gridSize = s->boxes->gridSize;
    int Zend = gridSize[2] - (gridSize[2] % 2);
    int Yend = gridSize[1] - (gridSize[1] % 2);

    //TODO: remember this needs to be done 4 times, shifted 
    for(int i=0; i < 2; i++) {
        for(int j=0; j < 2; j++) {
            for(int z=i; z < Zend; z += 2) {
                for(int y=j; y < Yend; y += 2) {
                    clusterForce(s, y, z);
                }
                if(Yend != gridSize[1]) {
                    clusterForce(s, Yend, z);
                }
            }
            if(Zend != gridSize[2]) {
                for(int y=0; y < Yend; y += 2) {
                    clusterForce(s, y, Zend);
                }
                if(Yend != gridSize[1]) {
                    clusterForce(s, Yend, Zend);
                }
            }
        }
    }
    ompReduceReal(reductionArray, s->boxes->nLocalBoxes, gridSize[0]);//TODO: remove extra tasks for 2x2 blocks

    real_t *ePotential = &(s->ePotential);
#pragma omp task depend(inout: reductionArray[0]) depend(out: ePotential[0])
    {
        startTimer(ePotReductionTimer);
        *ePotential = reductionArray[0]*4.0*((LjPotential*)(s->pot))->epsilon;
        reductionArray[0] = 0;
        stopTimer(ePotReductionTimer);
    }

    return 0;
}
