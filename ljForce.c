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


static double ePot_tp;
#pragma omp threadprivate (ePot_tp)
static real3* force_tp;
#pragma omp threadprivate (force_tp)


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


void calc_internal_force(int Box, int nJBox, SimFlat *s, int rCut2, int s6, int eShift, int epsilon)
{
    int nIBox = s->boxes->nAtoms[Box];
    // loop over atoms in Box
    for (int iOff=MAXATOMS*Box; iOff<(Box*MAXATOMS+nIBox); iOff++) {
        // loop over atoms in jBox
        for (int jOff=Box*MAXATOMS; jOff<(Box*MAXATOMS+nJBox); jOff++) {
            if (jOff <= iOff) //Can probably do a task special for local 
                continue; 
            real3 dr;
            real_t r2 = 0.0;
            for (int m=0; m<3; m++) {
                dr[m] = s->atoms->r[iOff][m] - s->atoms->r[jOff][m];//12 bn cycles
                r2+=dr[m]*dr[m];//5 bn cycles
            }
            if ( r2 <= rCut2 && r2 > 0.0) {//3 bn cycles
                // Important note:
                // from this point on r actually refers to 1.0/r
                r2 = 1.0/r2;
                real_t r6 = s6 * (r2*r2*r2);//3 bn cycles
                real_t eLocal = r6 * (r6 - 1.0) - eShift;
                s->atoms->U[iOff] += 0.5*eLocal;//2 bn cycles
                s->atoms->U[jOff] += 0.5*eLocal;
                if (Box < s->boxes->nLocalBoxes)
                    ePot_tp += eLocal;
                else
                    ePot_tp += 0.5*eLocal;

                // different formulation to avoid sqrt computation
                real_t fr = - 4.0*epsilon*r6*r2*(12.0*r6 - 6.0);
                for (int m=0; m<3; m++) {
                    force_tp[iOff][m] -= dr[m]*fr;
                    force_tp[jOff][m] += dr[m]*fr;
                }
            }
        } // loop over atoms in jBox
    } // loop over atoms in iBox
}

void calc_force(int iBox, int jBox, int nJBox, SimFlat *s, int rCut2, int s6, int eShift, int epsilon)
{
    int nIBox = s->boxes->nAtoms[iBox];
    // loop over atoms in iBox
    for (int iOff=MAXATOMS*iBox; iOff<(iBox*MAXATOMS+nIBox); iOff++) {
        // loop over atoms in jBox
        for (int jOff=jBox*MAXATOMS; jOff<(jBox*MAXATOMS+nJBox); jOff++) {
            real3 dr;
            real_t r2 = 0.0;
            for (int m=0; m<3; m++) {
                dr[m] = s->atoms->r[iOff][m] - s->atoms->r[jOff][m];//12 bn cycles
                r2+=dr[m]*dr[m];//5 bn cycles
            }

            if ( r2 <= rCut2 && r2 > 0.0) {//3 bn cycles
                // Important note:
                // from this point on r actually refers to 1.0/r
                r2 = 1.0/r2;
                real_t r6 = s6 * (r2*r2*r2);//3 bn cycles
                real_t eLocal = r6 * (r6 - 1.0) - eShift;
                s->atoms->U[iOff] += 0.5*eLocal;//2 bn cycles
                s->atoms->U[jOff] += 0.5*eLocal;
                if (jBox < s->boxes->nLocalBoxes)
                    ePot_tp += eLocal;
                else
                    ePot_tp += 0.5*eLocal;

                // different formulation to avoid sqrt computation
                real_t fr = - 4.0*epsilon*r6*r2*(12.0*r6 - 6.0);
                for (int m=0; m<3; m++) {
                    force_tp[iOff][m] -= dr[m]*fr;
                    force_tp[jOff][m] += dr[m]*fr;
                }
            }
        } // loop over atoms in jBox
    } // loop over atoms in iBox
}

int ljForce(SimFlat* s)
{
    LjPotential* pot = (LjPotential *) s->pot;
    real_t sigma = pot->sigma;
    real_t epsilon = pot->epsilon;
    real_t rCut = pot->cutoff;
    real_t rCut2 = rCut*rCut;

    real_t s6 = sigma*sigma*sigma*sigma*sigma*sigma;
    real_t rCut6 = s6 / (rCut2*rCut2*rCut2);
    real_t eShift = POT_SHIFT * rCut6 * (rCut6 - 1.0);
    int nNbrBoxes = 27;

    static real3* force = NULL;

    // zero forces and energy
    real_t ePot = 0.0;
    s->ePotential = 0.0;
    int fSize = s->boxes->nTotalBoxes*MAXATOMS;

#pragma omp parallel
    {
    if (force == NULL) {
#pragma omp single
        force = comdMalloc(fSize*omp_get_max_threads()*sizeof(real3));

        force_tp = force + fSize*omp_get_thread_num();
    }
    for(int ii=0; ii<fSize; ++ii)
        zeroReal3(force_tp[ii]);

#pragma omp for
    for (int ii=0; ii<fSize; ++ii) {
        zeroReal3(s->atoms->f[ii]);
        s->atoms->U[ii] = 0.;
    }
    ePot_tp = 0;
#pragma omp single
    {
    // loop over local boxes
    for (int iBox=0; iBox < s->boxes->nLocalBoxes; iBox++) {
        // loop over neighbors of iBox
        for (int jTmp=0; jTmp < nNbrBoxes; jTmp++) {
            int jBox  = s->boxes->nbrBoxes[iBox][jTmp];
            int nJBox = s->boxes->nAtoms[jBox];
            if (iBox > jBox) continue;
            if(iBox == jBox) {
#pragma omp task
                calc_internal_force(iBox, nJBox, s, rCut2, s6, eShift, epsilon);
            }
#pragma omp task 
            calc_force(iBox, jBox, nJBox, s, rCut2, s6, eShift, epsilon);

        } // loop over neighbor boxes
    } // loop over local boxes in system
    } //end single
#pragma omp taskwait
#pragma omp critical
        ePot += ePot_tp;
    } //end parallel

    // reduce thread private forces into s->atoms->f
#pragma omp parallel for
    for (int ii=0; ii<fSize; ++ii) {
        for (int jj=0; jj<omp_get_num_threads(); ++jj) {
            s->atoms->f[ii][0] += force[ii+fSize*jj][0];
            s->atoms->f[ii][1] += force[ii+fSize*jj][1];
            s->atoms->f[ii][2] += force[ii+fSize*jj][2];
        }
    }

    ePot = ePot*4.0*epsilon;
    s->ePotential = ePot;

    return 0;
}
