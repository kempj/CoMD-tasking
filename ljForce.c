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
void boxForce(int iBox, SimFlat *s)
{
    LjPotential* pot = (LjPotential *) s->pot;
    real_t rCut = pot->cutoff;
    real_t sigma = pot->sigma;
    real_t epsilon = pot->epsilon;
    real_t s6 = sigma*sigma*sigma*sigma*sigma*sigma;
    real_t rCut2 = rCut*rCut;
    real_t rCut6 = s6 / (rCut2*rCut2*rCut2);
    real_t eShift = POT_SHIFT * rCut6 * (rCut6 - 1.0);


    int nNbrBoxes = 27;
    int nIBox = s->boxes->nAtoms[iBox];
    for(int jTmp=0; jTmp < nNbrBoxes; jTmp++) {
        int jBox  = s->boxes->nbrBoxes[iBox][jTmp];
        int nJBox = s->boxes->nAtoms[jBox];
        for(int iOff=MAXATOMS*iBox; iOff<(iBox*MAXATOMS+nIBox); iOff++) {
            for(int jOff=jBox*MAXATOMS; jOff<(jBox*MAXATOMS+nJBox); jOff++) {
                real3 dr;
                real_t r2 = 0.0;
                for(int m=0; m<3; m++) {
                    dr[m] = s->atoms->r[iOff][m] - s->atoms->r[jOff][m];
                    r2+=dr[m]*dr[m];
                }
                if(r2<=rCut2 && r2>0.0) {
                    r2 = 1.0/r2;
                    real_t r6 = s6 * (r2*r2*r2);
                    real_t eLocal = r6 * (r6 - 1.0) - eShift;
                    s->atoms->U[iOff] += 0.5*eLocal;
                    reductionArray[iBox] += 0.5*eLocal;
                    //reductionArray[0] += 0.5*eLocal;

                    real_t fr = - 4.0*epsilon*r6*r2*(12.0*r6 - 6.0);
                    for (int m=0; m<3; m++) {
                        s->atoms->f[iOff][m] -= dr[m]*fr;
                    }
                }
            }
        }
    }
}

int ljForce(SimFlat* s)
{
    real3  *atomF = s->atoms->f;
    real3  *atomR = s->atoms->r;
    real_t *atomU = s->atoms->U;
    int neighbors[27];

    for (int iBox=0; iBox < s->boxes->nLocalBoxes; iBox++) {
        for(int nBox=0; nBox < 27; nBox++) {
            neighbors[nBox] =  s->boxes->nbrBoxes[iBox][nBox];
        }
#pragma omp task depend(out: atomU[iBox*MAXATOMS], reductionArray[iBox], atomF[iBox*MAXATOMS]) \
                 depend( in: atomR[neighbors[0 ]*MAXATOMS], atomR[neighbors[1 ]*MAXATOMS], atomR[neighbors[2 ]*MAXATOMS], \
                             atomR[neighbors[3 ]*MAXATOMS], atomR[neighbors[4 ]*MAXATOMS], atomR[neighbors[5 ]*MAXATOMS], \
                             atomR[neighbors[6 ]*MAXATOMS], atomR[neighbors[7 ]*MAXATOMS], atomR[neighbors[8 ]*MAXATOMS], \
                             atomR[neighbors[9 ]*MAXATOMS], atomR[neighbors[10]*MAXATOMS], atomR[neighbors[11]*MAXATOMS], \
                             atomR[neighbors[12]*MAXATOMS], atomR[neighbors[13]*MAXATOMS], atomR[neighbors[14]*MAXATOMS], \
                             atomR[neighbors[15]*MAXATOMS], atomR[neighbors[16]*MAXATOMS], atomR[neighbors[17]*MAXATOMS], \
                             atomR[neighbors[18]*MAXATOMS], atomR[neighbors[19]*MAXATOMS], atomR[neighbors[20]*MAXATOMS], \
                             atomR[neighbors[21]*MAXATOMS], atomR[neighbors[22]*MAXATOMS], atomR[neighbors[23]*MAXATOMS], \
                             atomR[neighbors[24]*MAXATOMS], atomR[neighbors[25]*MAXATOMS], atomR[neighbors[26]*MAXATOMS] )
        {
            for(int ii=iBox*MAXATOMS; ii<(iBox+1)*MAXATOMS;ii++) {
                zeroReal3(s->atoms->f[ii]);
                s->atoms->U[ii] = 0.;
            }
            //reductionArray[iBox] = 0.;// shouldn't be necessary
            boxForce(iBox, s);
        }
    }
    for (int iBox=s->boxes->nLocalBoxes; iBox < s->boxes->nTotalBoxes; iBox++) {
#pragma omp task depend(inout: atomU[iBox*MAXATOMS], atomF[iBox*MAXATOMS])
        for(int ii=iBox*MAXATOMS; ii<(iBox+1)*MAXATOMS;ii++) {
            zeroReal3(s->atoms->f[ii]);
            s->atoms->U[ii] = 0.;
        }
    }
    ompReduce(reductionArray, s->boxes->nTotalBoxes);

    real_t *ePotential = &(s->ePotential);
#pragma omp task depend(in: reductionArray[0]) depend(out: ePotential[0])
    *ePotential = reductionArray[0]*4.0*((LjPotential*)(s->pot))->epsilon;

    return 0;
}
