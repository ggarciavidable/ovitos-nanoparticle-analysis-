# -*- coding: utf-8 -*-
"""
OVITO modifier for estimating the transverse size of a nanoparticle during deformation.

Atomic positions are projected onto the xy plane and radial distances from the
projected nanoparticle centroid are calculated. The script reports the maximum
projected radius, the mean and standard deviation of the outermost 1% of radial
distances, and statistics for the uppermost and lowermost atomic z coordinates.

Developed and tested with OVITO 3.0.0-dev469. Compatibility with later OVITO
versions has not been systematically tested.

Run using OVITO's Python Script Modifier. The generated global attributes can be
exported through File -> Export File -> Table of Values.
"""
from ovito.data import *
import numpy as np
    
def modify(frame, data):
    particlePosition = data.particles.position
    xyCoords = particlePosition[:,:2]
    zCoords = particlePosition[:,2]
    centreOfMassXY = np.mean(xyCoords,axis=0)
    xyModulus = np.sqrt(np.sum(np.square(xyCoords-centreOfMassXY),1))
    maxR = np.max(xyModulus)
    xyModulus.sort()
    zCoords = np.sort(zCoords)
    xyModulus=xyModulus[::-1]
    meanDistance = np.mean( xyModulus[0:data.particles.count//100] )
    stdDistance = np.std( xyModulus[0:data.particles.count//100] )
    mean1pcFloorZ = np.mean(zCoords[0:data.particles.count//100])
    mean01pcFloorZ = np.mean(zCoords[0:data.particles.count//1000])
    stdZmin1pc = np.std(zCoords[0:data.particles.count//100] )
    stdZmin01pc = np.std(zCoords[0:data.particles.count//1000] )
    zCoords=zCoords[::-1]
    mean1pcTopZ = np.mean(zCoords[0:data.particles.count//100])
    mean01pcTopZ = np.mean(zCoords[0:data.particles.count//1000])
    stdZmax1pc = np.std(zCoords[0:data.particles.count//100] )
    stdZmax01pc = np.std(zCoords[0:data.particles.count//1000] )
    diffPerCent = (maxR-meanDistance)/maxR
    print("Std diff%:\n",stdDistance,diffPerCent)
    data.attributes['NP-eqrad_mean'] = meanDistance
    data.attributes['std-NP-eqrad'] = stdDistance
    data.attributes['NP-eqrad_max'] = maxR
    data.attributes['meanZTop1%'] = mean1pcTopZ
    data.attributes['stdZtop1%'] = stdZmax1pc
    data.attributes['meanZTop0.1%'] = mean01pcTopZ
    data.attributes['stdZTop0.1%'] = stdZmax01pc
    data.attributes['meanZFloor1%'] = mean1pcFloorZ
    data.attributes['stdZFloor1%'] = stdZmin1pc
    data.attributes['meanZFloor0.1%'] = mean01pcFloorZ
    data.attributes['stdZFloor0.1%'] = stdZmin01pc