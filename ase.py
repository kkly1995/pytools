# ase-based routines
import numpy as np
from ase.atoms import Atoms
from ase.units import Bohr, Hartree
from ase.calculators.singlepoint import SinglePointDFTCalculator
from typing import Union, Sequence, List
from os import mkdir, listdir

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
        type_map: list,
        ) -> None:
    """
    fname: str
        Name of file to write samples to. While any name is allowed,
        NEP (as of Oct 11, 2021) favors train.in and test.in.
    images: Atoms object, or list of Atoms objects
        The Atoms need not be identical in composition, number of atoms, etc.
        However, each object must have energy and forces.
        Not all Atoms need to have stress either (untested).
    type_map: list
        Like deepmd, NEP enumerates atoms starting from 0.
        This specifies which atoms are mapped to which number.
        Specifically, an atom with atomic number type_map[0] is mapped
        to 0, etc. The atomic numbers are parsed from
        get_atomic_numbers().
    """
    # file consists of two parts, header and body
    header = []
    body = []
    
    header.append('%s\n' % len(images))

    for image in images:
        types = image.get_atomic_numbers()
        cell = image.get_cell()[:]
        position = image.get_positions(wrap=True)
        energy = image.get_potential_energy()
        force = image.get_forces()
        # remap types
        new_types = np.full_like(types, -1)
        for i in range(len(type_map)):
            new_types[types == type_map[i]] = i
        if (new_types == -1).any():
            print('provided type map did not map all species')
            return
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
        line = '%d %.8f %.8f %.8f %.8f %.8f %.8f\n'
        for i in range(len(image)):
            body.append(line % (
                new_types[i],
                position[i,0], position[i,1], position[i,2],
                force[i,0], force[i,1], force[i,2]
                ))
    with open(fname, 'w') as f:
        f.writelines(header + body)

def read_nep(
        fname: str,
        type_map: list
        ) -> Union[Atoms, List[Atoms]]:
    """
    fname: str
        Name of file containing NEP data.
    type_map: list
        See type_map arg in write_nep. This routine will invert the mapping,
        i.e. 0 is mapped to type_map[0].
    """
    f = open(fname, 'r')
    # first line contains the number of samples
    N_samples = int(f.readline())
    # the next N_samples lines contains number of atoms per sample
    N = []
    for i in range(N_samples):
        line = f.readline()
        N.append(int(line.split()[0])) # ignore virial for now
    # begin reading the samples
    images = []
    for i in range(N_samples):
        # first line has the energy
        line = f.readline()
        energy = float(line.split()[0])
        # next line has the cell vectors
        cell = np.loadtxt(f, max_rows=1).reshape(3, 3)
        # next N[i] lines has type, positions, forces
        data = np.loadtxt(f, max_rows=N[i])
        types = data[:,0]
        positions = data[:,1:4]
        forces = data[:,4:]
        # remap types
        new_types =  np.full_like(types, -1)
        for j in range(len(type_map)):
            new_types[types==j] = type_map[j]
        if (new_types == -1).any():
            print('provided type map did not map all species')
            return
        # assemble Atoms image
        image = Atoms(
                numbers=new_types,
                positions=positions,
                cell=cell,
                pbc=True,
                )
        calc = SinglePointDFTCalculator(
                image,
                energy=energy,
                forces=forces,
                )
        image.calc = calc
        images.append(image)
    f.close()
    return images
