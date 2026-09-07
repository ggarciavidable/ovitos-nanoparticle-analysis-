"""
OVITO modifier for estimating strain and atomistic contact area during planar
compression of a nanoparticle.

The script reconstructs indentation depth and compressive strain and estimates
contact area using a Vergeles-type geometrical criterion. Relevant simulation
parameters are read, when available, from log.diamond or log.lammps located in
the same directory as the trajectory file.

Developed and tested with OVITO 3.0.0-dev469. Compatibility with later OVITO
versions has not been systematically tested. The log-file parsing rules and
several geometric parameters reflect the original simulation workflow and may
require adaptation for other setups.

Run using OVITO's Python Script Modifier. The generated global attributes can be
exported through File -> Export File -> Table of Values.
"""
from ovito.data import *
import numpy as np
from pathlib import Path
def modify(frame, data):
    #SIMULATION DATA (SET ACCORDING TO YOUR SIMULATION)
    topIndenterPosition = 104.841224142732  #change to initial top indenter position
    bottomIndenterPosition = -100.867687990085  #change to initial bottom indenter position
    initialTopIndenterSeparation = 5.0  #set to initial gap between indenter and top atoms
    initialFloorIndenterSeparation = 1.0  #set to initial gap between indenter and bottom atoms atoms
    indentationRate = 0.1  #set your indentation rate
    timestep = 0.001  #set the timestep of your simulation
    unloadStep = 512000  #set unload step

##########################################################################
    #THE FOLLOWING CODE READS OUT THE SIMULATION DATA FROM THE LOG FILE, 
    #COMMENT THESE LINES OUT IF YOU HAVE SET THE VALUES MANUALLY
    sourceFile = Path(data.attributes['SourceFile'])
    dumpDirectory = sourceFile.parent
    diamondLog = dumpDirectory / 'log.diamond'
    lammpsLog = dumpDirectory / 'log.lammps'

    if diamondLog.is_file():
        logFile = diamondLog
    elif lammpsLog.is_file():
        logFile = lammpsLog
    else:
        raise FileNotFoundError(
            "No 'log.diamond' or 'log.lammps' file was found in the trajectory directory: "
            f"{dumpDirectory}"
        )

    with logFile.open() as f :
        for l in f :
            if l.startswith('variable       rate equal ') :
                indentationRate = float(l.split()[3])
            elif l.startswith('timestep        ') :
                timestep = float(l.split()[1])
            elif l.startswith('variable       zmax0 equal ') and '${' not in l :
                topIndenterPosition = float(l.split()[-1])
            elif l.startswith('variable       zmin0 equal') and '${' not in l :
                bottomIndenterPosition = float(l.split()[-1])
            elif l.startswith('variable       zmax equal "c_zmax+') and '.' in l :
                initialTopIndenterSeparation = float(l.split()[-1].split('+')[-1][:-1])
            elif l.startswith('variable       zmin equal "c_zmin-') and '.' in l :
                initialFloorIndenterSeparation = float(l.split()[-1].split('-')[-1][:-1])
#########################################################################################
    ##CONSTANTS
    INIT_INDENTER_POS = topIndenterPosition  #INITIAL TOP INDENTER POSITION
    INIT_FLOOR_POS = bottomIndenterPosition  #INITIAL BOTTOM INDENTER POSITION
    TOP_INIT_GAP = initialTopIndenterSeparation  #INITIAL UPPER GAP BETWEEN SAMPLE AND TOP INDENTER
    FLOOR_INIT_GAP = initialFloorIndenterSeparation  #INITIAL LOWER GAP BETWEEN SAMPLE AND BOTTOM INDENTER
    D = INIT_INDENTER_POS-TOP_INIT_GAP-INIT_FLOOR_POS-FLOOR_INIT_GAP  #INITIAL DIAMETER OF THE SPHERICAL NP ALONG INDENTATION AXIS
    TIMESTEP = timestep  #SIMULATION TIMESTEP
    INDENTATION_RATE = indentationRate  #INDENTATION RATE
    #THE FOLLOWING TWO CONSTANTS ARE USED TO SELECT CONTACT AREA ATOMS AS THOSE THAT ARE SOME FRACTION "vergelesCutoff" OF THE CRYSTAL'S LATTICE PAREMETER a0 AWAY FROM THE TOP INDENTER
    # Operational parameters of the atomistic contact definition used in the
    # original diamond-nanoparticle simulations; adjust for other systems.
    a0 = 3.567  # lattice parameter
    vergelesCutoff = 0.25  # contact cutoff expressed as a fraction of a0
    CURRENT_TIMESTEP = data.attributes['Timestep']  #STORES THE STEP NUMBER OF THE CURRENT FRAME OF THE SIMULATION
    ##VERGELES AREA
    ###UPPER CONTACT AREA
    particlePosition = data.particles.position
    maxZ = np.max(particlePosition[:,2])
    minZ = np.min(particlePosition[:,2])
    indenterStep = CURRENT_TIMESTEP*TIMESTEP*INDENTATION_RATE
    indenterPosition = INIT_INDENTER_POS - indenterStep
    indenterLoadUnloadPosition = INIT_INDENTER_POS + (np.abs(unloadStep-CURRENT_TIMESTEP)-unloadStep)*INDENTATION_RATE*TIMESTEP
    contactAreaAtoms = particlePosition[particlePosition[:,2]>indenterPosition-vergelesCutoff*a0]
    Ntop = np.shape(contactAreaAtoms)[0]
    xyProjectionTop = contactAreaAtoms[:,:2] 
    centreOfMassXYTop = np.mean(xyProjectionTop,axis=0)
    minMaxStrain = (D-(maxZ-minZ))/D

    if Ntop!=0 :
        upperAreaVergeles = 2*np.pi*np.sum(np.power(xyProjectionTop-centreOfMassXYTop,2))/Ntop
        upperAreaVergelesRadius = np.sqrt(2*np.sum(np.power(xyProjectionTop-centreOfMassXYTop,2))/Ntop)
    else :
        upperAreaVergeles = 0.0
        upperAreaVergelesRadius = 0.0
    ###LOWER CONTACT AREA
    bottomAtoms = particlePosition[particlePosition[:,2]<INIT_FLOOR_POS+vergelesCutoff*a0]
    Nfloor = np.shape(bottomAtoms)[0]
    xyProjectionFloor = bottomAtoms[:,:2]
    centreOfMassXYFloor = np.mean(xyProjectionFloor,axis=0)
    if Nfloor!=0 :
        lowerAreaVergeles = 2*np.pi*np.sum(np.power(xyProjectionFloor-centreOfMassXYFloor,2))/Nfloor
    else :
        lowerAreaVergeles = 0.0
    
    ##INDENTATION DEPTH
    if indenterStep < TOP_INIT_GAP + FLOOR_INIT_GAP :
        h = 0
        H = 0
    else :
        h = indenterStep - TOP_INIT_GAP - FLOOR_INIT_GAP
        H = INIT_INDENTER_POS-indenterLoadUnloadPosition - TOP_INIT_GAP - FLOOR_INIT_GAP
        
    ##ANALYTICAL AREA EXPRESSION
    if h != 0 :
        analyticArea = np.pi*np.power((h/2)*np.sqrt((2*D/h) - 1),2)
        analyticRadius = (h/2)*np.sqrt((2*D/h) - 1)
    else :
        analyticArea = 0
        analyticRadius = 0
    ##ACCURATE STRAIN
    if indenterStep < TOP_INIT_GAP + FLOOR_INIT_GAP :
        strain = 0
        accStrain = 0
    else :
        strain = h/D
        accStrain = H/D
    if minMaxStrain<1e-5 :
        minMaxStrain = 0
    
    #OUTPUT GLOBAL VALUES
    data.attributes["upCAVergeles"] = upperAreaVergeles
    data.attributes["lowerContactAreaVergeles"] = lowerAreaVergeles
    data.attributes["indenterPosition"] = indenterPosition
    data.attributes["indentationDepth"] = h
    data.attributes["analyticArea"] = analyticArea
    data.attributes["strain"] = strain
    data.attributes["strain%"] = np.round(strain*100)
    data.attributes["realDiameter"] = D
    data.attributes["bottomIndenterPosition"] = INIT_FLOOR_POS
    data.attributes["a0"] =  a0
    data.attributes["VergelesCutoff"] = vergelesCutoff
    data.attributes["VRadius"] = upperAreaVergelesRadius 
    data.attributes["AnalyticRadius"] = analyticRadius
    data.attributes["ZMAX"] = maxZ
    data.attributes["ZMIN"] = minZ
    data.attributes["strainFromLoadUnloadFormula"] = accStrain
    data.attributes["min-max-Strain"] = minMaxStrain
    
    #SCREEN OUTPUT
    print("Depths:\n h H\n",h,H)
    print("strain from loading formula\n",strain)
    print("strain from load-unload formula\n",accStrain)
    print("min-max-Strain\n",minMaxStrain)
    print("indenter Position",indenterPosition)
    print("Upper contact area\n",upperAreaVergeles)
    print("Max Z coord\n",maxZ)
    print(upperAreaVergelesRadius)