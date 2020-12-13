import sys
sys.path.append('/Users/santy/odrive/Google Drive/github projects/questaal-reader')
from questaal_reader import get_bands
from pymatgen.electronic_structure.plotter import BSPlotter as bp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
ry2ev = 13.605662285137

def get_weights(bnds_file="data/lifeo2/bnds_color.temp"):
    """Gets the band structure contribution from bnds file

    Args:
        bnds_file (str, optional): bands file. Defaults to "data/lifeo2/bnds_color.temp".

    Returns:
        array: Li contributions
    """    
    with open(bnds_file, 'r') as f:
        lines = f.read().splitlines()
    neval=int(lines[0].split()[0])
    # ncol=np.sum([1 if "col" in i  else 0 for i in lines[0].split()])
    ncol=int(lines[0].split()[2])
    nkpts=float(lines[1])
    kpts=[list(map(float,i.split())) for i in lines[2:] if "   " in i and len(i.split())==3]
    kpts=np.array(kpts)
    kpts_positions=[j+3 for j,i in enumerate(lines[2:]) if "   " in i and len(i.split())==3]
    from itertools import groupby
    kpts_rep,_=[(len(list(values)), key) for key, values in groupby(kpts.T[0])][0]
    nsp=  int(kpts_rep/(2*ncol)) if ncol!=0 else int(kpts_rep)
    energies=[]
    for i in kpts_positions:
        vals=[]
        tmp=0
        while len(vals)<=neval:
            vals=vals+list(map(float,lines[i+tmp].split()))
            tmp+=1
        energies.append(vals[:neval])
    energies=np.array(energies)*ry2ev

    Li_spin1=energies[1::4].T
    Li_spin2=energies[3::4].T
    return Li_spin1,Li_spin2

def plot_Li(ax,folder="data/lifeo2/"):
    
    scale=2.1e-2
    Li_spin1,Li_spin2=get_weights(bnds_file=folder+"bnds_color.temp")
    p=get_bands(folder+"bnds.temp")
    bp_plot=bp(p).bs_plot_data()
    d=bp_plot['distances']
    d=np.array(d).flatten()
    e_up=bp_plot['energy']['1']
    e_down=bp_plot['energy']['-1']
    for nbnd in range(np.array(e_up).shape[2]):
        energy_up=np.array(e_up)[:,nbnd,:].flatten()
        ax.plot(d,energy_up,c="#d62828",lw=1.3,label="spin $\\uparrow$")
        
        energy_down=np.array(e_down)[:,nbnd,:].flatten()
        ax.plot(d,energy_down,c="#003049",ls="-",lw=1.3,label="spin $\\downarrow$")

        # ax.scatter(d,energy_down,Li_spin2[nbnd]*scale,facecolor="none",edgecolor="#f94144")
        # ax.scatter(d,energy_up,Li_spin1[nbnd]*scale,facecolor="none",edgecolor="#f94144")
        ax.fill_between(d,energy_down-Li_spin2[nbnd]*scale,energy_down+Li_spin2[nbnd]*scale,color="#e9c46a")
        ax.fill_between(d,energy_up-Li_spin1[nbnd]*scale,energy_up+Li_spin1[nbnd]*scale,color="#e9c46a",label="Li")
        
    for j,i in enumerate(bp_plot['ticks']["distance"]):
            ax.axvline(i,c="k",ls="--",lw=1)
    for j,i in enumerate ([ax]):
        fs=13
        i.tick_params(axis='y',labelsize=fs)
        i.set_xticks(bp_plot['ticks']["distance"])
        i.set_xticklabels(bp_plot['ticks']["label"],fontsize=fs)
        i.axhline(0,ls="-.",c="k",lw=1)
        i.set_ylim(-5,5)
        i.autoscale(enable=True, axis='x', tight=True)
        i.legend()
        handles, labels = i.get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        i.legend(handles, labels, loc='lower left')