# ase-based routines
import numpy as np
from ase.io import read, write
from ase.atoms import Atoms
from ase.units import Bohr, Hartree
from ase.calculators.singlepoint import SinglePointDFTCalculator
from typing import Union, Sequence, List, Any
from os import mkdir, listdir

def read_ipi_xyz(
        fname: str,
        positions: str,
        cell: str,
        index: Any = None
        ):
    """
    Read xyz configuration(s) used by i-PI.
    At the moment I don't know why the comment line in i-PI differs from
    ASE convention, perhaps there is an option I am missing.

    This uses ASE's reader but goes through the file an extra time to correct
    for the cell, which it cannot (?) parse.

    fname: str
        Name of file to read.
    positions: 'A' or 'B'
        Whether the positions are given in angstroms (A) or bohr (B)
    cell: 'A' or 'B'
        Whether the cell parameters, namely the lengths, are in angstroms
        or bohr
    index: either None or ':'
        Same as index in ase.io.read(), i.e. will be passed in as index=index
        However, this routine will only handle two cases:
        either the file contains only one image,
        in which case None should be passed here
        or the file contains more than one,
        in which case ':' should be passed here.

        The reason for this is to make it easy to match cell parameters
        upon rereading.
    """
    image = read(fname, format='extxyz', index=index)
    if isinstance(image, list):
        N = len(image)
        # change to angstrom if given in bohr
        if positions == 'B':
            for i in range(N):
                r = image[i].get_positions()*Bohr
                image[i].set_positions(r)
        # reread and get cell parameters
        f = open(fname, 'r')
        for i in range(N):
            next(f) # skip first line, which is just len(image[i])
            line = f.readline()
            word = line.split()
            cellpar = np.zeros(6)
            for j in range(6):
                cellpar[j] = float(word[j+2])
            # sucks putting this conditional in the loop but whatever
            if cell == 'B':
                cellpar[:3] *= Bohr
            image[i].set_cell(cellpar)
            image[i].set_pbc(True)
            # skip position lines
            for _ in range(len(image[i])):
                next(f)
        f.close()
    else:
        if positions == 'B':
            r = image.get_positions()*Bohr
            image.set_positions(r)
        with open(fname, 'r') as f:
            next(f) # skips first line?
            line = f.readline()
        word = line.split()
        cellpar = np.zeros(6)
        for i in range(6):
            cellpar[i] = float(word[i+2])
        if cell == 'B':
            cellpar[:3] *= Bohr
        image.set_cell(cellpar)
        image.set_pbc(True)
    return image

def write_ipi_xyz(
        fname: str,
        image: Atoms,
        fmt: str,
        ):
    """
    uses ASE's writer but changes the comment line to one that i-PI likes
    currently only for writing a single sample
    """
    write(fname, image, format='extxyz')
    cellpar = image.cell.cellpar()
    comment = '# CELL(abcABC): '
    for i in range(6):
        comment += fmt % cellpar[i] + ' '
    comment += 'positions{angstrom} cell{angstrom}\n'
    with open(fname, 'r') as f:
        lines = f.readlines()
    lines[1] = comment
    with open(fname, 'w') as f:
        f.writelines(lines)

def write_deepmd(
        path: str,
        images: Union[Atoms, Sequence[Atoms]],
        type_map: list,
        write_virial: bool = False,
        subsets: int = 1
        ) -> None:
    """
    Data will be written using ASE's default units,
    which is also consistent with deepmd datasets.

    path: str
        Instead of specifying a file or filename to write to,
        a path must be specified since the deepmd format uses multiple files.
        The actual filenames will follow the recommendations that can be found
        in the deepmd documentation, i.e. the files that will be created are
        path/set.000/coord.npy, path/set.000/energy.npy, etc.
    images: Atoms object or list of Atoms objects
        Currently allows single Atoms object,
        though ideally since one is constructing a dataset
        there should really be more than one object.

        All of these images should have the same number of atoms
        and the same atom types. They must also have energies and forces,
        as obtained from the calculator property.
    type_map: list
        Since deepmd enumerates types starting from 0, this specifies
        which atoms are mapped to which number.
        Specifically, an atom with atomic number type_map[0] is mapped
        to 0. The atomic numbers are parsed from
        get_atomic_numbers().
    write_virial: bool
        While this routine strictly requires energies and forces,
        it does not require virials / stresses.
        If True, virials will be written to path/set.000/virial.npy, etc.
    subsets: int
        Splits images into subsets, which will then be put into
        path/set.000/, path/set.001/, etc.
        Requires the total number of images to be divisible by the number
        of subsets.
    """
    if isinstance(images, Atoms):
        images = [images]
        # in ASE this step is actually handled independent of the format
        # but i include it here for now
    num_images = len(images)
    if num_images == 1:
        # print warning but allow this, at least for the moment
        print("Warning: writing only a single sample")
    if num_images % subsets != 0:
        msg = '%s images were to be split into %s subsets, ' % (num_images, subsets)
        msg += 'but %s is not divisible by %s' % (num_images, subsets)
        print(msg)
        return
    images_per_set = num_images // subsets
    box = []
    coord = []
    types = []
    energy = []
    force = []
    virial = [] # will only be used if necessary
    for i in range(num_images):
        box.append(images[i].get_cell()[:].flatten())
        coord.append(images[i].get_positions(wrap=True).flatten())
        types.append(images[i].get_atomic_numbers())
        energy.append(images[i].get_potential_energy())
        force.append(images[i].get_forces().flatten())
        if write_virial:
            virial.append(-images[i].get_volume() * \
                    images[i].get_stress(voigt=False).flatten())
        # final check on types
        if not np.allclose(types[i], types[0]):
            print('change in atom types detected in image %s' % i)
            return
    types = types[0] # since types across samples are consistent
    # map types
    new_types = np.full_like(types, -1)
    for i in range(len(type_map)):
        new_types[types == type_map[i]] = i
    if (new_types == -1).any():
        print('provided type map did not map all species')
        return
    # if the number of atoms changes, it will be detected here,
    # when converting to arrays
    box = np.array(box)
    coord = np.array(coord)
    types = np.array(types)
    energy = np.array(energy)
    force = np.array(force)
    virial = np.array(virial) # works even if empty
    # split into subsets
    box = np.reshape(box, (subsets, images_per_set, -1))
    coord = np.reshape(coord, (subsets, images_per_set, -1))
    energy = np.reshape(energy, (subsets, images_per_set))
    force = np.reshape(force, (subsets, images_per_set, -1))
    if write_virial:
        virial = np.reshape(virial, (subsets, images_per_set, -1))
    # attempt write
    mkdir(path)
    np.savetxt(path + '/type.raw', new_types, fmt='%d')
    for s in range(subsets):
        directory = path + '/set.' + str(s).zfill(3)
        mkdir(directory)
        np.save(directory + '/box', box[s])
        np.save(directory + '/coord', coord[s])
        np.save(directory + '/energy', energy[s])
        np.save(directory + '/force', force[s])
        if write_virial:
            np.save(directory + '/virial', virial[s])

def read_deepmd(
        path: str,
        type_map: list,
        read_virial: bool = False,
        ) -> Union[Atoms, List[Atoms]]:
    """
    Assumes the data is written according to deepmd's conventions,
    i.e. it comes in sets path/set.000/, path/set.001/, etc.
    and within each set the filenames are box.npy, coord.npy, etc.

    path: str
        Name of path containing deepmd data.
    type_map: list
        Converts types from type.raw into atomic numbers for ASE.
        Species i (as read from type.raw) will be given atomic number
        type_map[i].
    """
    # read and map types
    types = np.loadtxt(path + '/type.raw')
    new_types = np.full_like(types, -1)
    for i in range(len(type_map)):
        new_types[types==i] = type_map[i]
    if (new_types == -1).any():
        print('provided type map did not map all species')
        return
    # determine the number of sets
    subsets = 0
    for name in listdir(path):
        try:
            set_number = int(name.split('.')[1]) + 1
            if set_number > subsets:
                subsets = set_number
        except:
            # if this fails, that just means name is not set.nnn
            # which is fine (other stuff can be in the directory)
            continue
    # if subsets is still 0 at this point, that means no sets were found
    if subsets == 0:
        print('no sets were found in ' + path)
        return
    # now start looking inside these sets
    N = len(types)
    images = []
    if read_virial:
        for s in range(subsets):
            cell = np.load(path + '/set.' + str(s).zfill(3) + '/box.npy')
            positions = np.load(path + '/set.' + str(s).zfill(3) + '/coord.npy')
            energy = np.load(path + '/set.' + str(s).zfill(3) + '/energy.npy')
            forces = np.load(path + '/set.' + str(s).zfill(3) + '/force.npy')
            virial = np.load(path + '/set.' + str(s).zfill(3) + '/virial.npy')
            num_images = cell.shape[0]
            cell = cell.reshape(num_images, 3, 3)
            positions = positions.reshape(num_images, N, 3)
            forces = forces.reshape(num_images, N, 3)
            # the calculator needs stress
            volume = np.linalg.det(cell)
            stress = -virial / volume[:,np.newaxis]
            stress = stress.reshape(num_images, 3, 3)
            for i in range(num_images):
                image = Atoms(
                        numbers=new_types,
                        positions=positions[i],
                        cell=cell[i],
                        pbc=True,
                        )
                calc = SinglePointDFTCalculator(
                        image,
                        energy=energy[i],
                        forces=forces[i],
                        stress=stress[i],
                        )
                image.calc = calc
                images.append(image)
        return images
    else:
        for s in range(subsets):
            cell = np.load(path + '/set.' + str(s).zfill(3) + '/box.npy')
            positions = np.load(path + '/set.' + str(s).zfill(3) + '/coord.npy')
            energy = np.load(path + '/set.' + str(s).zfill(3) + '/energy.npy')
            forces = np.load(path + '/set.' + str(s).zfill(3) + '/force.npy')
            num_images = cell.shape[0]
            cell = cell.reshape(num_images, 3, 3)
            positions = positions.reshape(num_images, N, 3)
            forces = forces.reshape(num_images, N, 3)
            for i in range(num_images):
                image = Atoms(
                        numbers=new_types,
                        positions=positions[i],
                        cell=cell[i],
                        pbc=True,
                        )
                calc = SinglePointDFTCalculator(
                        image,
                        energy=energy[i],
                        forces=forces[i],
                        )
                image.calc = calc
                images.append(image)
        return images

def write_runner(
        fname: str,
        images: Union[Atoms, Sequence[Atoms]],
        ) -> None:
    """
    Currently: no comments and no charges. Data will be written in
    RuNNer's preferred units, Bohr and Hartree.

    fname: str
        Name of file to write to. Note that RuNNer favors
        the name 'input.data'; however, any name is allowed here
        just in case.
    images: Atoms object, or list of Atoms objects
        The Atoms need not be identical in composition, number of atoms, etc.
        However, each object must have energy and forces.
    """
    if isinstance(images, Atoms):
        images = [images]
    lines = []
    for image in images:
        N = len(image)
        r = image.get_positions(wrap=True) / Bohr
        cell = image.get_cell()[:] / Bohr
        force = image.get_forces() * (Bohr / Hartree)
        types = image.get_chemical_symbols()
        energy = image.get_potential_energy() / Hartree
        # begin writing
        lines.append('begin\n')
        # lattice vectors
        line = 'lattice {:>12.6f} {:>12.6f} {:>12.6f}\n'
        for j in range(3):
            lines.append(line.format(cell[j,0], cell[j,1], cell[j,2]))
        # atoms and forces
        line = 'atom {:>12.8f} {:>12.8f} {:>12.8f} {:>3} 0.0 0.0 '
        line += '{:>12.8f} {:>12.8f} {:>12.8f}\n'
        for j in range(N):
            lines.append(line.format(r[j,0], r[j,1], r[j,2], types[j],\
                    force[j,0], force[j,1], force[j,2]))
        lines.append('energy %.8f\n' % energy)
        lines.append('charge 0.0\n')
        lines.append('end\n')
    with open(fname, 'w') as f:
        f.writelines(lines)

def write_nep(
        fname: str,
        images: Union[Atoms, Sequence[Atoms]],
        ) -> None:
    """
    fname: str
        Name of file to write samples to. While any name is allowed,
        NEP (as of Oct 11, 2021) favors train.in and test.in.
    images: Atoms object, or list of Atoms objects
        The Atoms need not be identical in composition, number of atoms, etc.
        However, each object must have energy and forces.
        Not all Atoms need to have stress either (untested).
    """
    # file consists of two parts, header and body
    header = []
    body = []
    
    header.append('%s\n' % len(images))

    for image in images:
        symbols = image.get_chemical_symbols()
        cell = image.get_cell()[:]
        position = image.get_positions(wrap=True)
        energy = image.get_potential_energy()
        force = image.get_forces()
        # figure out if there's virial info
        virial = np.zeros(6)
        has_virial = False
        try:
            virial = -image.get_stress()*image.get_volume()
            has_virial = True
        except:
            # keep has_virial false
            pass
        # start writing
        header.append('%d %d\n' % (len(image), has_virial))
        # energy and virial line
        if has_virial:
            body.append('%.8f %.8f %.8f %.8f %.8f %.8f %.8f\n' % (
                energy,
                virial[0],
                virial[1],
                virial[2],
                virial[5], # off diagonals do not follow ASE's order
                virial[3],
                virial[4]
                ))
        else:
            body.append('%.8f\n' % energy)
        # cell line
        body.append('%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\n' % (
            cell[0,0], cell[0,1], cell[0,2],
            cell[1,0], cell[1,1], cell[1,2],
            cell[2,0], cell[2,1], cell[2,2]
            ))
        # atom lines
        line = '%s %.8f %.8f %.8f %.8f %.8f %.8f\n'
        for i in range(len(image)):
            body.append(line % (
                symbols[i],
                position[i,0], position[i,1], position[i,2],
                force[i,0], force[i,1], force[i,2]
                ))
    with open(fname, 'w') as f:
        f.writelines(header + body)

def read_nep(
        fname: str
        ) -> Union[Atoms, List[Atoms]]:
    """
    fname: str
        Name of file containing NEP data.
    """
    f = open(fname, 'r')
    # first line contains the number of samples
    N_samples = int(f.readline())
    # the next N_samples lines contains number of atoms per sample
    N = []
    has_virial = []
    for i in range(N_samples):
        line = f.readline()
        word = line.split()
        N.append(int(word[0]))
        has_virial.append(bool(int(word[1])))
    # begin reading the samples
    images = []
    for i in range(N_samples):
        # first line has the energy
        line = f.readline()
        word = line.split()
        energy = float(word[0])
        virial = []
        if has_virial[i]:
            for j in range(6):
                virial.append(float(word[j+1]))
            stress = np.array([\
                    [virial[0], virial[3], virial[5]],\
                    [virial[3], virial[1], virial[4]],\
                    [virial[5], virial[4], virial[2]],\
                    ])
        # next line has the cell vectors
        cell = np.loadtxt(f, max_rows=1).reshape(3, 3)
        if has_virial[i]:
            # convert stress
            stress /= -np.linalg.det(cell)
        # next N[i] lines has type, positions, forces
        symbols = []
        positions = []
        forces = []
        for _ in range(N[i]):
            line = f.readline()
            word = line.split()
            symbols.append(word[0])
            positions.append([float(word[1]), float(word[2]), float(word[3])])
            forces.append([float(word[4]), float(word[5]), float(word[6])])
        positions = np.array(positions)
        forces = np.array(forces)
        # assemble Atoms image
        image = Atoms(
                symbols=symbols,
                positions=positions,
                cell=cell,
                pbc=True,
                )
        if has_virial[i]:
            calc = SinglePointDFTCalculator(
                    image,
                    energy=energy,
                    forces=forces,
                    stress=stress,
                    )
            image.calc = calc
        else:
            calc = SinglePointDFTCalculator(
                    image,
                    energy=energy,
                    forces=forces,
                    )
            image.calc = calc
        images.append(image)
    f.close()
    return images
