# Visual-SELEX
We developed a rapid-response program called Visual-SELEX. This program generates stable 3D structures of ssDNA using coarse-grained simulations and MD simulations. It then transforms these structures into point cloud models. Finally, Visual-SELEX identifies similar spatial structures of ssDNA obtained after the SELEX process by performing point cloud model registration and superposition analysis.
## Installation
Ubuntu >=22.04
Python >=3.7
PyTorch >=1.1
numpy	>= 1.18.1
vermouth	>= 0.7.1
networkx	~=2.0
scipy	>= 1.5.4
decorator	4.4.2
tqdm	>= 4.43.0
numba >= 0.51.2
Polyply =1.0
Pyuul =0.2.0
## Running Example
### 1. We used the Polyply tool to construct a coarse-grained model for ssDNA
```
polyply gen_params -lib martini2 -o complexDNA.itp -name ssDNA -seqf DNA.fasta
polyply gen_coords -p martini2.top -b build_file_martini.bld -name DNA -dens 1000 -o martini2.gro
```
### 2. Molecular dynamics simulations utilizing Gromacs
```
gmx grompp -f minim.mdp -c martini2.gro -p martini2.top -o 01-minim.tpr -maxwarn 1
gmx grompp -f equil.mdp -c 01-minim.gro -r 01-minim.gro -p martini2.top -o 02-equil.tpr -maxwarn 2
gmx grompp -f mdrun.mdp -c 02-equil.gro -r 02-equil.gro -t 02-equil.cpt -p martini2.top -o 03-mdrun.tpr -maxwarn 2
gmx grompp -f md.mdp -c 03-mdrun.gro -t 03-mdrun.cpt -p martini2.top -o 04-md.tpr -maxwarn 2
```
### 3. Transforming coarse-grained ssDNA into an all-atom representation of ssDNA
```
polyply gen_coords -p full.top -mc bk.gro -box 22.44751 22.44751 22.44751 -name backmapped -b full_build.bld -o cg_full_DNA.gro -back_fudge 0.8
```
### 4. Generate point cloud model using Pyuul
### Importing libraries
```
from pyuul import VolumeMaker # the main PyUUL module
from pyuul import utils # the PyUUL utility module
import time,os,urllib # some standard python modules we are going to use
```
### Parsing the structures
```
coords, atname = utils.parsePDB("exampleStructures/") # get coordinates and atom names
atoms_channel = utils.atomlistToChannels(atname) # calculates the corresponding channel of each atom
radius = utils.atomlistToRadius(atname) # calculates the radius of each atom
```
### Volumetric representations
```
SurfacePoitCloud = PointCloudSurfaceObject(coords, radius)
VolumePoitCloud = PointCloudVolumeObject(coords, radius)
VoxelRepresentation = VoxelsObject(coords, radius, atoms_channel)
```
### 5. Point cloud registration
```
python reg.py
```
### 6. Point cloud overlap
```
python overlap.py --ovlap
```
