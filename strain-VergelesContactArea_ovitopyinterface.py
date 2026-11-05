"""
OVITO modifier for estimating strain and atomistic contact area during planar
compression of a nanoparticle.

The script reconstructs indentation depth and compressive strain and estimates
contact area using the relation proposed by Vergeles et al. for the lateral
distribution of atoms identified as being in contact with the indenter.

Relevant simulation parameters are read, when available, from log.diamond or
log.lammps located in the same directory as the trajectory file. The parser
recognizes the variable names used in the original simulation workflow.

Developed and tested with OVITO 3.0.0-dev469. Compatibility with later OVITO
versions has not been systematically tested.

Loading-unloading simulations are supported when `unloadStep` is set manually
to the timestep at which the indenter reverses direction. This assumes an
immediate reversal, with no hold segment. Set `unloadStep = None` for monotonic
loading simulations.

Run using OVITO's Python Script Modifier. The generated global attributes can
be exported through File -> Export File -> Table of Values.
"""

from pathlib import Path

import numpy as np
from ovito.data import *


def modify(frame, data):
    # -------------------------------------------------------------------------
    # Simulation parameters
    # Defaults correspond to the original workflow and are overwritten by
    # values parsed from log.diamond/log.lammps when matching definitions exist.
    # -------------------------------------------------------------------------
    topIndenterPosition = 104.841224142732
    bottomIndenterPosition = -100.867687990085
    initialTopIndenterSeparation = 5.0
    initialFloorIndenterSeparation = 1.0
    indentationRate = 0.1
    timestep = 0.001

    # Set to an integer timestep for immediate load -> unload reversal.
    # Leave as None for monotonic loading. Hold segments are not represented.
    unloadStep = None

    # Operational parameters of the contact definition used in the
    # diamond-nanoparticle simulations.
    a0 = 3.567
    vergelesCutoff = 0.25

    # -------------------------------------------------------------------------
    # Read simulation parameters from the log file
    # -------------------------------------------------------------------------
    sourceFile = Path(data.attributes["SourceFile"])
    dumpDirectory = sourceFile.parent
    diamondLog = dumpDirectory / "log.diamond"
    lammpsLog = dumpDirectory / "log.lammps"

    if diamondLog.is_file():
        logFile = diamondLog
    elif lammpsLog.is_file():
        logFile = lammpsLog
    else:
        raise FileNotFoundError(
            "No 'log.diamond' or 'log.lammps' file was found in the trajectory "
            f"directory: {dumpDirectory}"
        )

    with logFile.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            tokens = stripped.split()

            if len(tokens) >= 4 and tokens[:3] == ["variable", "rate", "equal"]:
                indentationRate = float(tokens[3])

            elif tokens and tokens[0] == "timestep" and len(tokens) >= 2:
                timestep = float(tokens[1])

            elif len(tokens) >= 4 and tokens[:3] == ["variable", "zmax0", "equal"]:
                value = tokens[3]
                if "${" not in value:
                    try:
                        topIndenterPosition = float(value)
                    except ValueError:
                        pass

            elif len(tokens) >= 4 and tokens[:3] == ["variable", "zmin0", "equal"]:
                value = tokens[3]
                if "${" not in value:
                    try:
                        bottomIndenterPosition = float(value)
                    except ValueError:
                        pass

            elif len(tokens) >= 4 and tokens[:3] == ["variable", "zmax", "equal"]:
                expression = tokens[3].strip('"')
                if expression.startswith("c_zmax+"):
                    try:
                        initialTopIndenterSeparation = float(
                            expression.split("+", 1)[1]
                        )
                    except ValueError:
                        pass

            elif len(tokens) >= 4 and tokens[:3] == ["variable", "zmin", "equal"]:
                expression = tokens[3].strip('"')
                if expression.startswith("c_zmin-"):
                    try:
                        initialFloorIndenterSeparation = float(
                            expression.split("-", 1)[1]
                        )
                    except ValueError:
                        pass

    # -------------------------------------------------------------------------
    # Geometry and indenter motion
    # -------------------------------------------------------------------------
    INIT_INDENTER_POS = topIndenterPosition
    INIT_FLOOR_POS = bottomIndenterPosition
    TOP_INIT_GAP = initialTopIndenterSeparation
    FLOOR_INIT_GAP = initialFloorIndenterSeparation

    # Initial nanoparticle extent along the loading axis.
    D = (
        INIT_INDENTER_POS
        - TOP_INIT_GAP
        - INIT_FLOOR_POS
        - FLOOR_INIT_GAP
    )

    CURRENT_TIMESTEP = data.attributes["Timestep"]

    loadingDisplacement = CURRENT_TIMESTEP * timestep * indentationRate

    if unloadStep is None:
        actualDisplacement = loadingDisplacement
    else:
        # Immediate reversal at unloadStep, with the same speed on unloading.
        if CURRENT_TIMESTEP <= unloadStep:
            actualDisplacement = loadingDisplacement
        else:
            displacementAtReversal = unloadStep * timestep * indentationRate
            actualDisplacement = (
                displacementAtReversal
                - (CURRENT_TIMESTEP - unloadStep) * timestep * indentationRate
            )

    indenterPosition = INIT_INDENTER_POS - actualDisplacement

    # -------------------------------------------------------------------------
    # Atomistic contact area
    # -------------------------------------------------------------------------
    particlePosition = data.particles.position
    maxZ = np.max(particlePosition[:, 2])
    minZ = np.min(particlePosition[:, 2])

    contactAreaAtoms = particlePosition[
        particlePosition[:, 2] > indenterPosition - vergelesCutoff * a0
    ]
    Ntop = np.shape(contactAreaAtoms)[0]

    if Ntop != 0:
        xyProjectionTop = contactAreaAtoms[:, :2]
        centreOfMassXYTop = np.mean(xyProjectionTop, axis=0)
        squaredRadialDistances = np.sum(
            np.power(xyProjectionTop - centreOfMassXYTop, 2), axis=1
        )
        upperAreaVergeles = 2 * np.pi * np.sum(squaredRadialDistances) / Ntop
        upperAreaVergelesRadius = np.sqrt(
            2 * np.sum(squaredRadialDistances) / Ntop
        )
    else:
        upperAreaVergeles = 0.0
        upperAreaVergelesRadius = 0.0

    bottomAtoms = particlePosition[
        particlePosition[:, 2] < INIT_FLOOR_POS + vergelesCutoff * a0
    ]
    Nfloor = np.shape(bottomAtoms)[0]

    if Nfloor != 0:
        xyProjectionFloor = bottomAtoms[:, :2]
        centreOfMassXYFloor = np.mean(xyProjectionFloor, axis=0)
        squaredFloorDistances = np.sum(
            np.power(xyProjectionFloor - centreOfMassXYFloor, 2), axis=1
        )
        lowerAreaVergeles = 2 * np.pi * np.sum(squaredFloorDistances) / Nfloor
    else:
        lowerAreaVergeles = 0.0

    # -------------------------------------------------------------------------
    # Strain and indentation depth
    # -------------------------------------------------------------------------
    totalInitialGap = TOP_INIT_GAP + FLOOR_INIT_GAP

    if actualDisplacement < totalInitialGap:
        indentationDepth = 0.0
        strain = 0.0
    else:
        indentationDepth = actualDisplacement - totalInitialGap
        strain = indentationDepth / D

    # Loading-only displacement/strain retained as diagnostic quantities.
    if loadingDisplacement < totalInitialGap:
        loadingIndentationDepth = 0.0
        loadingStrain = 0.0
    else:
        loadingIndentationDepth = loadingDisplacement - totalInitialGap
        loadingStrain = loadingIndentationDepth / D

    minMaxStrain = (D - (maxZ - minZ)) / D
    if minMaxStrain < 1e-5:
        minMaxStrain = 0.0

    # Analytical projected area for an ideal sphere compressed symmetrically.
    if indentationDepth > 0.0 and indentationDepth < 2 * D:
        analyticRadius = (
            indentationDepth
            / 2
            * np.sqrt((2 * D / indentationDepth) - 1)
        )
        analyticArea = np.pi * analyticRadius**2
    else:
        analyticArea = 0.0
        analyticRadius = 0.0

    # -------------------------------------------------------------------------
    # OVITO global attributes
    # -------------------------------------------------------------------------
    data.attributes["upCAVergeles"] = upperAreaVergeles
    data.attributes["lowerContactAreaVergeles"] = lowerAreaVergeles
    data.attributes["indenterPosition"] = indenterPosition
    data.attributes["indentationDepth"] = indentationDepth
    data.attributes["analyticArea"] = analyticArea
    data.attributes["strain"] = strain
    data.attributes["strain%"] = strain * 100.0
    data.attributes["realDiameter"] = D
    data.attributes["bottomIndenterPosition"] = INIT_FLOOR_POS
    data.attributes["a0"] = a0
    data.attributes["VergelesCutoff"] = vergelesCutoff
    data.attributes["VRadius"] = upperAreaVergelesRadius
    data.attributes["AnalyticRadius"] = analyticRadius
    data.attributes["ZMAX"] = maxZ
    data.attributes["ZMIN"] = minZ
    data.attributes["loadingStrain"] = loadingStrain
    data.attributes["min-max-Strain"] = minMaxStrain

    # Backward-compatible name used by earlier versions of this script.
    data.attributes["strainFromLoadUnloadFormula"] = strain

    # -------------------------------------------------------------------------
    # Screen output
    # -------------------------------------------------------------------------
    print("Indentation depth:", indentationDepth)
    print("Strain:", strain)
    print("Loading-only strain:", loadingStrain)
    print("Min-max strain:", minMaxStrain)
    print("Indenter position:", indenterPosition)
    print("Upper contact area:", upperAreaVergeles)
    print("Max Z coordinate:", maxZ)
    print("Vergeles contact radius:", upperAreaVergelesRadius)
