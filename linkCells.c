/// \file
/// Functions to maintain link cell structures for fast pair finding.
///
/// In CoMD 1.1, atoms are stored in link cells.  Link cells are widely
/// used in classical MD to avoid an O(N^2) search for atoms that
/// interact.  Link cells are formed by subdividing the local spatial
/// domain with a Cartesian grid where the grid spacing in each
/// direction is at least as big as he potential's cutoff distance.
/// Because atoms don't interact beyond the potential cutoff, for an
/// atom iAtom in any given link cell, we can be certain that all atoms
/// that interact with iAtom are contained in the same link cell, or one
/// of the 26 neighboring link cells.
/// 
/// CoMD chooses the link cell size (boxSize) on each axis to be the
/// shortest possible distance, longer than cutoff, such that the local
/// domain size divided by boxSize is an integer.  I.e., the link cells
/// are commensurate with with the local domain size.  While this does
/// not result in the smallest possible link cells, it does allow us to
/// keep a strict separation between the link cells that are entirely
/// inside the local domain and those that represent halo regions.
///
/// The number of local link cells in each direction is stored in
/// gridSize.  Local link cells have 3D grid coordinates (ix, iy, iz)
/// where ix, iy, and iz can range from 0 to gridSize[iAxis]-1,
/// whiere iAxis is 0 for x, 1 for y and 2 for the z direction.  The
/// number of local link cells is thus nLocalBoxes =
/// gridSize[0]*gridSize[1]*gridSize[2].
///
/// The local link cells are surrounded by one complete shell of halo
/// link cells.  The halo cells provide temporary storage for halo or
/// "ghost" atoms that belong to other tasks, but whose coordinates are
/// needed locally to complete the force calculation.  Halo link cells
/// have at least one coordinate with a value of either -1 or
/// gridSize[iAxis].
///
/// Because CoMD stores data in ordinary 1D C arrays, a mapping is
/// needed from the 3D grid coords to a 1D array index.  For the local
/// cells we use the conventional mapping ix + iy*nx + iz*nx*ny.  This
/// keeps all of the local cells in a contiguous region of memory
/// starting from the beginning of any relevant array and makes it easy
/// to iterate the local cells in a single loop.  Halo cells are mapped
/// differently.  After the local cells, the two planes of link cells
/// that are face neighbors with local cells across the -x or +x axis
/// are next.  These are followed by face neighbors across the -y and +y
/// axis (including cells that are y-face neighbors with an x-plane of
/// halo cells), followed by all remaining cells in the -z and +z planes
/// of halo cells.  The total number of link cells (on each rank) is
/// nTotalBoxes.
///
/// Data storage arrays that are used in association with link cells
/// should be allocated to store nTotalBoxes*MAXATOMS items.  Data for
/// the first atom in linkCell iBox is stored at index iBox*MAXATOMS.
/// Data for subsequent atoms in the same link cell are stored
/// sequentially, and the number of atoms in link cell iBox is
/// nAtoms[iBox].
///
/// \see getBoxFromTuple is the 3D->1D mapping for link cell indices.
/// \see getTuple is the 1D->3D mapping
///
/// \param [in] cutoff The cutoff distance of the potential.

#include "linkCells.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "parallel.h"
#include "memUtils.h"
#include "decomposition.h"
#include "performanceTimers.h"
#include "CoMDTypes.h"

#define   MIN(A,B) ((A) < (B) ? (A) : (B))
#define   MAX(A,B) ((A) > (B) ? (A) : (B))

static void copyAtom(Atoms* in, Atoms* out, int iAtom, int iBox, int jAtom, int jBox);
static void emptyHaloCells(LinkCell* boxes);

LinkCell* initLinkCells(const Domain* domain, real_t cutoff)
{
    assert(domain);
    LinkCell* ll = comdMalloc(sizeof(LinkCell));

    for (int i = 0; i < 3; i++) {
        ll->localMin[i] = domain->localMin[i];
        ll->localMax[i] = domain->localMax[i];
        ll->gridSize[i] = domain->localExtent[i] / cutoff; // local number of boxes
        ll->boxSize[i] = domain->localExtent[i] / ((real_t) ll->gridSize[i]);
        ll->invBoxSize[i] = 1.0/ll->boxSize[i];
    }

    ll->nLocalBoxes = ll->gridSize[0] * ll->gridSize[1] * ll->gridSize[2];

    ll->nHaloBoxes = 2 * ((ll->gridSize[0] + 2) *
            (ll->gridSize[1] + ll->gridSize[2] + 2) +
            (ll->gridSize[1] * ll->gridSize[2]));

    ll->nTotalBoxes = ll->nLocalBoxes + ll->nHaloBoxes;

    ll->nAtoms = comdMalloc(ll->nTotalBoxes*sizeof(int));
    for (int iBox=0; iBox<ll->nTotalBoxes; ++iBox) {
        ll->nAtoms[iBox] = 0;
    }

    assert ( (ll->gridSize[0] >= 2) && (ll->gridSize[1] >= 2) && (ll->gridSize[2] >= 2) );

    ll->nbrBoxes = comdMalloc(ll->nTotalBoxes*sizeof(int*));
    for (int iBox=0; iBox<ll->nTotalBoxes; ++iBox) {
        ll->nbrBoxes[iBox] = comdMalloc(27*sizeof(int));
    }

    for(int iBox=0; iBox<ll->nLocalBoxes; ++iBox) {
        getLocalNeighborBoxes(ll, iBox, ll->nbrBoxes[iBox]);
    }

    return ll;
}

void destroyLinkCells(LinkCell** boxes)
{
    if (! boxes) return;
    if (! *boxes) return;

    comdFree((*boxes)->nAtoms);
    comdFree(*boxes);
    *boxes = NULL;

    return;
}

//for shared memory only, takes a tuple for a halo cell and returns the local cell that it corresponds to.
void haloToLocalCell(int *x, int *y, int *z, int *gridSize)
{
    if(*x == -1 ) 
        *x = gridSize[0]-1;            
    if(*y == -1)
        *y = gridSize[1]-1;
    if(*z == -1)
        *z = gridSize[2]-1;
    if(*x == gridSize[0])
        *x = 0;
    if(*y == gridSize[1])
        *y = 0;
    if(*z == gridSize[2])
        *z = 0;
}

//for shared memory only, takes a box ID for a halo cell and returns the local cell that it corresponds to.
int getLocalHaloTuple(LinkCell *boxes, int iBox) {
    int x,y,z;
    getTuple(boxes, iBox, &x, &y, &z);

    haloToLocalCell(&x, &y, &z, boxes->gridSize);

    return getBoxFromTuple(boxes, x, y, z);
}

int getLocalNeighborBoxes(LinkCell* boxes, int iBox, int* nbrBoxes)
{
    int ix, iy, iz;
    getTuple(boxes, iBox, &ix, &iy, &iz);

    int count = 0;
    for (int i=ix-1; i<=ix+1; i++) {
        for (int j=iy-1; j<=iy+1; j++) {
            for (int k=iz-1; k<=iz+1; k++) {
                nbrBoxes[count++] = getLocalHaloTuple(boxes, getBoxFromTuple(boxes,i,j,k));
            }
        }
    }
    return count;
}


/// \details
/// Populates the nbrBoxes array with the 27 boxes that are adjacent to
/// iBox.  The count is 27 instead of 26 because iBox is included in the
/// list (as neighbor 13).  Caller is responsible to alloc and free
/// nbrBoxes.
/// \return The number of nbr boxes (always 27 in this implementation).

//TODO: This shouldn't be used any more.
int getNeighborBoxes(LinkCell* boxes, int iBox, int* nbrBoxes)
{
    int ix, iy, iz;
    getTuple(boxes, iBox, &ix, &iy, &iz);

    int count = 0;
    for (int i=ix-1; i<=ix+1; i++) {
        for (int j=iy-1; j<=iy+1; j++) {
            for (int k=iz-1; k<=iz+1; k++) {
                nbrBoxes[count++] = getBoxFromTuple(boxes,i,j,k);
            }
        }
    }
    return count;
}

int getHaloNeighborBoxes(LinkCell* boxes, int iBox, int* nbrBoxes)
{
    int ix, iy, iz;
    getTuple(boxes, iBox, &ix, &iy, &iz);
    //const real_t* localMin = boxes->localMin; // alias
    //const real_t* localMax = boxes->localMax; // alias
    const int* gridSize = boxes->gridSize; // alias

    int count = 0;
    for (int i=ix-1; i<=ix+1; i++) {
        for (int j=iy-1; j<=iy+1; j++) {
            for (int k=iz-1; k<=iz+1; k++) {
                if(i < 0 || i >= gridSize[0] || 
                        j < 0 || j >= gridSize[1] ||
                        k < 0 || k >= gridSize[2] ) {
                    nbrBoxes[count] = iBox;
                } else {
                    nbrBoxes[count] = getBoxFromTuple(boxes, i, j, k);
                }
                count++;
            }
        }
    }
    
    return count;
}


/// \details
/// Finds the appropriate link cell for an atom based on the spatial
/// coordinates and stores data in that link cell.
/// \param [in] gid   The global of the atom.
/// \param [in] iType The species index of the atom.
/// \param [in] x     The x-coordinate of the atom.
/// \param [in] y     The y-coordinate of the atom.
/// \param [in] z     The z-coordinate of the atom.
/// \param [in] px    The x-component of the atom's momentum.
/// \param [in] py    The y-component of the atom's momentum.
/// \param [in] pz    The z-component of the atom's momentum.
void putAtomInBox(LinkCell* boxes, Atoms* atoms,
        const int gid, const int iType,
        const real_t x,  const real_t y,  const real_t z,
        const real_t px, const real_t py, const real_t pz)
{
    real_t xyz[3] = {x,y,z};

    // Find correct box.
    int iBox = getBoxFromCoord(boxes, xyz);
    int iOff = iBox*MAXATOMS;
    iOff += boxes->nAtoms[iBox];

    // assign values to array elements
    if (iBox < boxes->nLocalBoxes)
        atoms->nLocal++;
    boxes->nAtoms[iBox]++;

    atoms->gid[iOff] = gid;
    atoms->iSpecies[iOff] = iType;

    atoms->r[iOff][0] = x;
    atoms->r[iOff][1] = y;
    atoms->r[iOff][2] = z;

    atoms->p[iOff][0] = px;
    atoms->p[iOff][1] = py;
    atoms->p[iOff][2] = pz;
}

/// Calculates the link cell index from the grid coords.  The valid
/// coordinate range in direction ii is [-1, gridSize[ii]].  Any
/// coordinate that involves a -1 or gridSize[ii] is a halo link cell.
/// Because of the order in which the local and halo link cells are
/// stored the indices of the halo cells are special cases.
/// \see initLinkCells for an explanation of storage order.
int getBoxFromTuple(LinkCell* boxes, int ix, int iy, int iz)
{
    int iBox = 0;
    const int* gridSize = boxes->gridSize; // alias

    if (iz == gridSize[2]) {
        // Halo in Z+
        iBox = boxes->nLocalBoxes + 2*gridSize[2]*gridSize[1] + 2*gridSize[2]*(gridSize[0]+2) +
            (gridSize[0]+2)*(gridSize[1]+2) + (gridSize[0]+2)*(iy+1) + (ix+1);
    } else if (iz == -1) {
        // Halo in Z-
        iBox = boxes->nLocalBoxes + 2*gridSize[2]*gridSize[1] + 2*gridSize[2]*(gridSize[0]+2) +
            (gridSize[0]+2)*(iy+1) + (ix+1);
    } else if (iy == gridSize[1]) {
        // Halo in Y+
        iBox = boxes->nLocalBoxes + 2*gridSize[2]*gridSize[1] + gridSize[2]*(gridSize[0]+2) +
            (gridSize[0]+2)*iz + (ix+1);
    } else if (iy == -1) {
        // Halo in Y-
        iBox = boxes->nLocalBoxes + 2*gridSize[2]*gridSize[1] + iz*(gridSize[0]+2) + (ix+1);
    } else if (ix == gridSize[0]) {
        // Halo in X+
        iBox = boxes->nLocalBoxes + gridSize[1]*gridSize[2] + iz*gridSize[1] + iy;
    } else if (ix == -1) {
        // Halo in X-
        iBox = boxes->nLocalBoxes + iz*gridSize[1] + iy;
    } else {
        // local link celll.
        iBox = ix + gridSize[0]*iy + gridSize[0]*gridSize[1]*iz;
    }
    assert(iBox >= 0);
    assert(iBox < boxes->nTotalBoxes);

    return iBox;
}

/// Move an atom from one link cell to another.
/// \param srcPosition [in]  The index with box iBox of the atom to be moved.
/// \param srcBox [in] The index of the link cell the particle is moving from.
/// \param destBox [in] The index of the link cell the particle is moving to.
void moveAtom( LinkCell* srcBoxes, LinkCell *destBoxes, 
               Atoms* srcAtoms, Atoms* destAtoms, 
               int srcPosition, int srcBox, int destBox)
{
    int destPosition = destBoxes->nAtoms[destBox];
    copyAtom(srcAtoms, destAtoms, srcPosition, srcBox, destPosition, destBox);
    destBoxes->nAtoms[destBox]++;

    assert(destBoxes->nAtoms[destBox] < MAXATOMS);

    srcBoxes->nAtoms[srcBox]--;

    int ni = srcBoxes->nAtoms[srcBox];
    //This fills the 'hole' with the last entry in the cell.
    if (ni) {
        copyAtom(srcAtoms, srcAtoms, ni, srcBox, srcPosition, srcBox);
    }
    if (destBox > destBoxes->nLocalBoxes) {
        --destAtoms->nLocal;
        --srcAtoms->nLocal;
    }
    return;
}

int nextHighestAtom(Atoms *sourceAtoms, int atomOffset, int nAtoms, int maxGID, int minGID) 
{
    int targetPosition = 0;
    int targetGID = maxGID;
    for(int i = 0; i < nAtoms; i++) {
        if(sourceAtoms->gid[atomOffset+i] > minGID && sourceAtoms->gid[atomOffset+i] <= targetGID) {
            targetPosition = i;
            targetGID = sourceAtoms->gid[atomOffset+i];
        }
    }
    //printf("next highest GID (after %d) is %d in position %d\n", minGID, targetGID, targetPosition);
    return targetPosition;
}

//This copies a cell from one box buffer to another.
void copySortedCell(LinkCell *sourceBoxes, LinkCell *destBoxes, Atoms *sourceAtoms, Atoms *destAtoms, int iBox)
{
    const int numAtoms = sourceBoxes->nAtoms[iBox];
    int highestGID, lowestGID;
    int atomOffset = MAXATOMS * iBox;
    highestGID = lowestGID = sourceAtoms->gid[atomOffset];
    for(int i = atomOffset; i < atomOffset + numAtoms; i++) {
        if(sourceAtoms->gid[i] < lowestGID) {
            lowestGID = sourceAtoms->gid[i];
        }
        if(sourceAtoms->gid[i] > highestGID) {
            highestGID = sourceAtoms->gid[i];
        }
    }

    //printf("\ncopying cell %d with GIDs from %d to %d\n\n", iBox, lowestGID, highestGID);

    int targetAtom = nextHighestAtom(sourceAtoms, atomOffset, numAtoms, highestGID, lowestGID - 1);
    //printf("\n");
    for(int atomNum = 0; atomNum < numAtoms; atomNum++) {
        //printf("writing Atom %d (gid = %d) to position %d\n", atomNum, sourceAtoms->gid[atomOffset + atomNum], targetAtom);
        copyAtom(sourceAtoms, destAtoms, atomNum, iBox, targetAtom, iBox);
        targetAtom = nextHighestAtom(sourceAtoms, atomOffset, numAtoms, highestGID, sourceAtoms->gid[atomOffset + targetAtom]);
        
    }
    destBoxes->nAtoms[iBox] = sourceBoxes->nAtoms[iBox];
}

//The correctness of the task dependencies here depends on the assumption that there are no
//dependencies between this function and the function that last wrote the position
void updateLinkCells(LinkCell* boxes, LinkCell* boxesBuffer, Atoms* atoms, Atoms* atomsBuffer)
{
    real3  *atomF = atoms->f;
    real3  *atomR = atoms->r;
    real_t *atomU = atoms->U;
    real3  *atomP = atoms->p;
    real3  *atomsBufferR = atomsBuffer->r;

    int neighbors[27];
    for(int iBox=0; iBox<boxes->nLocalBoxes; iBox++) {
        for(int nBox=0; nBox < 27; nBox++) {
            neighbors[nBox] = boxes->nbrBoxes[iBox][nBox];
        }
#pragma omp task depend(out: atomsBufferR[iBox*MAXATOMS]) \
                 depend( in: atomR[neighbors[0 ]*MAXATOMS], atomR[neighbors[1 ]*MAXATOMS], atomR[neighbors[2 ]*MAXATOMS], \
                             atomR[neighbors[3 ]*MAXATOMS], atomR[neighbors[4 ]*MAXATOMS], atomR[neighbors[5 ]*MAXATOMS], \
                             atomR[neighbors[6 ]*MAXATOMS], atomR[neighbors[7 ]*MAXATOMS], atomR[neighbors[8 ]*MAXATOMS], \
                             atomR[neighbors[9 ]*MAXATOMS], atomR[neighbors[10]*MAXATOMS], atomR[neighbors[11]*MAXATOMS], \
                             atomR[neighbors[12]*MAXATOMS], atomR[neighbors[13]*MAXATOMS], atomR[neighbors[14]*MAXATOMS], \
                             atomR[neighbors[15]*MAXATOMS], atomR[neighbors[16]*MAXATOMS], atomR[neighbors[17]*MAXATOMS], \
                             atomR[neighbors[18]*MAXATOMS], atomR[neighbors[19]*MAXATOMS], atomR[neighbors[20]*MAXATOMS], \
                             atomR[neighbors[21]*MAXATOMS], atomR[neighbors[22]*MAXATOMS], atomR[neighbors[23]*MAXATOMS], \
                             atomR[neighbors[24]*MAXATOMS], atomR[neighbors[25]*MAXATOMS], atomR[neighbors[26]*MAXATOMS] )
        {   //This task pulls all atoms that belong in cell iBox from multiple cells in the main buffer to cell iBox in the secondary buffer
            boxesBuffer->nAtoms[iBox] = 0;
            int firstCopy = 0;
            for(int i=0; i<27; i++) {
                int neighborBox = boxes->nbrBoxes[iBox][i];
                if(neighborBox != iBox || firstCopy == 0) {
                    if(neighborBox == iBox)
                        firstCopy = 1;
                    for(int atomNum = 0; atomNum < boxes->nAtoms[neighborBox]; atomNum++) {
                        real3 tempPosition;
                        for(int i=0; i<3; i++) {
                            tempPosition[i] = atoms->r[neighborBox*MAXATOMS + atomNum][i];
                            if(tempPosition[i] >= boxes->localMax[i]) {
                                tempPosition[i] -= boxes->localMax[i];
                            } else if(tempPosition[i] <= boxes->localMin[i]) {
                                tempPosition[i] += boxes->localMax[i];
                            }
                        }
                        if(iBox == getBoxFromCoord(boxes, tempPosition)) {
                            copyAtom(atoms, atomsBuffer, atomNum, neighborBox, boxesBuffer->nAtoms[iBox], iBox);
                            for(int i=0; i<3; i++) {
                                atomsBuffer->r[iBox*MAXATOMS + boxesBuffer->nAtoms[iBox]][i] = tempPosition[i];
                            }
                            boxesBuffer->nAtoms[iBox]++;
                        }
                    }
                }
            }
        }
    }

    //At this point, the buffer is the correct output for the iteration.
    //TODO: set up some sort of pointer swap so the tasks that just copy back aren't necessary.

    //This loop copies the cells from the buffer back to the main buffer.
    for(int iBox=0; iBox<boxes->nLocalBoxes; ++iBox) {
#pragma omp task depend(in : atomsBufferR[iBox*MAXATOMS]) \
                 depend(out: atomF[iBox*MAXATOMS], atomR[iBox*MAXATOMS],\
                             atomU[iBox*MAXATOMS], atomP[iBox*MAXATOMS])
        copySortedCell(boxesBuffer, boxes, atomsBuffer, atoms, iBox);
    }
}

/// \return The largest number of atoms in any link cell.
int maxOccupancy(LinkCell* boxes)
{
    int localMax = 0;
    for (int ii=0; ii<boxes->nLocalBoxes; ++ii)
        localMax = MAX(localMax, boxes->nAtoms[ii]);

    int globalMax;

    startTimer(commReduceTimer);
    maxIntParallel(&localMax, &globalMax, 1);
    stopTimer(commReduceTimer);

    return globalMax;
}

/// Copy atom iAtom in link cell iBox to atom jAtom in link cell jBox.
/// Any data at jAtom, jBox is overwritten.  This routine can be used to
/// re-order atoms within a link cell.
void copyAtom(Atoms* in, Atoms* out, int inAtom, int inBox, int outAtom, int outBox)
{
    const int inOff = MAXATOMS*inBox+inAtom;
    const int outOff = MAXATOMS*outBox+outAtom;
    out->gid[outOff] = in->gid[inOff];
    out->iSpecies[outOff] = in->iSpecies[inOff];
    memcpy(out->r[outOff], in->r[inOff], sizeof(real3));
    memcpy(out->p[outOff], in->p[inOff], sizeof(real3));
    memcpy(out->f[outOff], in->f[inOff], sizeof(real3));
    memcpy(out->U+outOff,  in->U+inOff,  sizeof(real_t));
}

/// Get the index of the link cell that contains the specified
/// coordinate.  This can be either a halo or a local link cell.
///
/// Because the rank ownership of an atom is strictly determined by the
/// atom's position, we need to take care that all ranks will agree which
/// rank owns an atom.  The conditionals at the end of this function are
/// special care to ensure that all ranks make compatible link cell
/// assignments for atoms that are near a link cell boundaries.  If no
/// ranks claim an atom in a local cell it will be lost.  If multiple
/// ranks claim an atom it will be duplicated.
int getBoxFromCoord(LinkCell* boxes, real_t rr[3])
{
    const real_t* localMin = boxes->localMin; // alias
    const real_t* localMax = boxes->localMax; // alias
    const int*    gridSize = boxes->gridSize; // alias
    int ix = (int)(floor((rr[0] - localMin[0])*boxes->invBoxSize[0]));
    int iy = (int)(floor((rr[1] - localMin[1])*boxes->invBoxSize[1]));
    int iz = (int)(floor((rr[2] - localMin[2])*boxes->invBoxSize[2]));

    // For each axis, if we are inside the local domain, make sure we get
    // a local link cell.  Otherwise, make sure we get a halo link cell.
    if(rr[0] < localMax[0]) {
        if (ix == gridSize[0]) {
            ix = gridSize[0] - 1;
        }
    } else {
        ix = gridSize[0]; // assign to halo cell
    }
    if(rr[1] < localMax[1]) {
        if (iy == gridSize[1]) {
            iy = gridSize[1] - 1;
        }
    } else {
        iy = gridSize[1];
    }
    if(rr[2] < localMax[2]) {
        if (iz == gridSize[2]) {
            iz = gridSize[2] - 1;
        }
    } else {
        iz = gridSize[2];
    }

    return getBoxFromTuple(boxes, ix, iy, iz);
}

/// Set the number of atoms to zero in all halo link cells.
void emptyHaloCells(LinkCell* boxes)
{
    for (int ii=boxes->nLocalBoxes; ii<boxes->nTotalBoxes; ++ii)
        boxes->nAtoms[ii] = 0;
}

/// Get the grid coordinates of the link cell with index iBox.  Local
/// cells are easy as they use a standard 1D->3D mapping.  Halo cell are
/// special cases.
/// \see initLinkCells for information on link cell order.
/// \param [in]  iBox Index to link cell for which tuple is needed.
/// \param [out] ixp  x grid coord of link cell.
/// \param [out] iyp  y grid coord of link cell.
/// \param [out] izp  z grid coord of link cell.
void getTuple(LinkCell* boxes, int iBox, int* ixp, int* iyp, int* izp)
{
    int ix, iy, iz;
    const int* gridSize = boxes->gridSize;

    // If a local box
    if( iBox < boxes->nLocalBoxes) {
        ix = iBox % gridSize[0];
        iBox /= gridSize[0];
        iy = iBox % gridSize[1];
        iz = iBox / gridSize[1];
    } else {  // It's a halo box
        int ink = iBox - boxes->nLocalBoxes;
        if(ink < 2*gridSize[1]*gridSize[2]) {
            if (ink < gridSize[1]*gridSize[2]) {
                ix = 0;
            } else {
                ink -= gridSize[1]*gridSize[2];
                ix = gridSize[0] + 1;
            }
            iy = 1 + ink % gridSize[1];
            iz = 1 + ink / gridSize[1];
        } else if (ink < (2 * gridSize[2] * (gridSize[1] + gridSize[0] + 2))) {
            ink -= 2 * gridSize[2] * gridSize[1];
            if (ink < ((gridSize[0] + 2) *gridSize[2])) {
                iy = 0;
            } else {
                ink -= (gridSize[0] + 2) * gridSize[2];
                iy = gridSize[1] + 1;
            }
            ix = ink % (gridSize[0] + 2);
            iz = 1 + ink / (gridSize[0] + 2);
        } else {
            ink -= 2 * gridSize[2] * (gridSize[1] + gridSize[0] + 2);
            if (ink < ((gridSize[0] + 2) * (gridSize[1] + 2))) {
                iz = 0;
            } else {
                ink -= (gridSize[0] + 2) * (gridSize[1] + 2);
                iz = gridSize[2] + 1;
            }
            ix = ink % (gridSize[0] + 2);
            iy = ink / (gridSize[0] + 2);
        }
        // Calculated as off by 1
        ix--;
        iy--;
        iz--;
    }

    *ixp = ix;
    *iyp = iy;
    *izp = iz;
}


