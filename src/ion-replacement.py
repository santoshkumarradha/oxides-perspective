import os
from contextlib import contextmanager
from pymatgen.io.ase import AseAtomsAdaptor as p2ase
from pymatgen.io.cif import CifParser, CifWriter
from lmf import *
import pickle
import time
import numpy as np
from pymatgen import Structure, Lattice
from pathlib import Path
import sys
import os
from path import Path as cpath
sys.path.append(os.getcwd())


def get_structure(d=10, a=0, atom="Li"):
    species = []
    coords = []
    # does not work with cluster
    # parser = CifParser(
    #     "/home/srr70/coo2/other_compoun/group_1_atoms/test/licoo2.cif")
    # structure = parser.get_structures()[0]
    # does not work with cluster
    structure = Structure.from_file(
        "/home/srr70/coo2/other_compoun/group_1_atoms/test/licoo2.cif")
    for i in structure:
        if i.species_string == "Li":
            species.append(i.species)
            coords.append(i.coords + [0, 0, +.5 * d + a])
        else:
            species.append(i.species)
            coords.append(i.coords)
    a = structure.lattice.matrix.copy()
    a[2][2] = a[2][2] - d
    struc = Structure(Lattice(a), species, coords, coords_are_cartesian=True)
    struc.replace_species({"Li": atom})
    CifWriter(struc).write_file(
        f"{atom}"+'coo2_'+str(struc.lattice.c)[:4]+'.cif')
    #     CifWriter(struc).write_file(fname+'licoo2_big.cif')
    #     os.system("open "+fname+'licoo2_big.cif')
    return p2ase().get_atoms(struc)


def do_calc_atom(atom, d):
    import time
    start_time = time.time()
    fname = "./" + str(atom)+"/" + f"distance_{d}/"
    Path(fname).mkdir(parents=True, exist_ok=True)
    print(f"running atom={atom} d={d}")
    atoms = get_structure(d=d, atom=atom)
    with cpath(fname):
        calculator = lmf(nkabc=[4, 4, 4], ctrl="temp", p=num_processors)
        pot_energy = calculator.get_potential_energy(atoms)
        # lmf().clean()

    # ---time calculations
    elapsed_time = time.time() - start_time
    total_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    program_time_ellapsed_unformated = time.time() - program_start_time
    program_time_ellapsed = time.strftime(
        "%H:%M:%S", time.gmtime(program_time_ellapsed_unformated))

    # ---load the structure and energy to completed dictonray
    print("completed in {} total time {}\n =================\n".format(
        total_time, program_time_ellapsed))
    return pot_energy


# ----------------variables--------
num_processors = 12
program_start_time = time.time()


values = {}
for atom in ["Li", "Na", "K", "Rb", "Cs"]:
    for d in np.linspace(0, 5, 10):
        try:
            energy = do_calc_atom(atom=atom, d=d)
        except:
            print(f"Unable to do {atom} {d}")
            energy = 0
        values[atom+f'-{d}'] = energy

with open('data.pickle', 'wb') as handle:
    pickle.dump(values, handle, protocol=pickle.HIGHEST_PROTOCOL)
