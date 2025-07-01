import sisl
from hubbard import HubbardHamiltonian, sp2, density, plot, NEGF
import ase
from ase.visualize import view
import tk
import matplotlib.pyplot as plt
from ase.build import graphene_nanoribbon
import numpy as np
import netCDF4
from sisl import geom
import os

def init_V(H,V,elec_indx,axis=2):
    L=max(H.geometry[:,axis])-min(H.geometry[:,axis])
    elec_idx_L = []
    elec_idx_R = []
    for ia in np.ravel(elec_indx):
        if ia < 0:
            elec_idx_L.append(range(H.na)[ia])
        else:
            elec_idx_R.append(ia)
            
    for atom_i in np.arange(0,H.geometry.na):
        pos=H.geometry[atom_i,axis]
        if H.dim==1:
            if atom_i in elec_idx_L:
                H.H[atom_i,atom_i]=H.H[atom_i,atom_i] - 0.5*V
            elif atom_i in elec_idx_R:
                H.H[atom_i,atom_i]=H.H[atom_i,atom_i] + 0.5*V
            else:
               H.H[atom_i,atom_i]=H.H[atom_i,atom_i] - V*(pos-0.5*L)/L
        elif H.dim==2:
            if atom_i in elec_idx_L:
                H.H[atom_i,atom_i,0]=H.H[atom_i,atom_i,0]-0.5*V
                H.H[atom_i,atom_i,1]=H.H[atom_i,atom_i,1]- 0.5*V
            elif atom_i in elec_idx_R:
                H.H[atom_i,atom_i,0]=H.H[atom_i,atom_i,0]+ 0.5*V
                H.H[atom_i,atom_i,1]=H.H[atom_i,atom_i,1]+ 0.5*V
            else:
                H.H[atom_i,atom_i,0]=H.H[atom_i,atom_i,0] - V*(pos-0.5*L)/L
                H.H[atom_i,atom_i,1]=H.H[atom_i,atom_i,1] - V*(pos-0.5*L)/L

V=1
gate=0.125
kx=1
U=3.5
met='tf'
ltf=3
C_self = 0.1
plot_dq=False

newpath = f'./results_{V}V_{gate}e_{U}U_{met}_{ltf}lambda_{C_self}Ci_{kx}kx' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

# Read Device
device=sisl.get_sile('device.fdf').read_geometry()
# Remove H from device
device=device.remove(device.atoms.Z==1)
device.set_nsc([3,1,1])

# Read Electrode
elec=sisl.get_sile('elec.xyz').read_geometry()
elec.set_nsc([3,1,3])
elec_ts=elec.tile(2,axis=2)

# Ungated Electrode
H0 = sp2(elec_ts, t1=2.7, t2=0.0, t3=0.0, spin='polarized')
q0=H0.q0
dn=gate/q0
print(dn,gate)

###################Electrode Calculation ################################
# Build sisl.Hamiltonian object using the sp2 function
H_elec = sp2(elec_ts, t1=2.7, t2=0.0, t3=0.0, spin='polarized',dq=dn)
q=H_elec.q0
# Build the HubbardHamiltonian object with U=3. eV
MFH_elec = HubbardHamiltonian(H_elec, U=U, nkpt=[30,1,30], q=(0.5*q, 0.5*q),kT=0.025)

# set initial spin density
MFH_elec.random_density()

# converge 
MFH_elec.converge(density.calc_n, tol=1e-6, print_info=True, steps=10)
MFH_elec.shift(MFH_elec.fermi_level())

# Plot
fig, ax = plt.subplots(figsize=(2,8),)
bs = sisl.BandStructure(MFH_elec.H, [[0.5, 0, 0],[0, 0, 0], [0.0, 0, 0.5]], 100,['X', r'Gamma', 'Z'])
lk = bs.lineark(ticks=False)
bs_eig = bs.apply.array.eigh()
plt.plot(lk, bs_eig,c='k')
plt.ylim([-4,4])
plt.ylabel('E (eV)', size=14)
plt.xlabel('k', size=14)
ax.fill_between(lk, y1=-10, y2=0,color='gray',alpha=0.25)
plt.tight_layout()
fig.savefig(f'{newpath}/Electrode_Bandstructure.png',dpi=200)
plt.clf()

###################Device Calculation (EQ)################################
dQ=gate*8
dN=dQ/device.na

# Build sisl.Hamiltonian object using the sp2 function
HC= sp2(device, t1=2.7, t2=0.0, t3=0.0, spin='polarized',dq=dN)
# with periodic boundary conditions
HC.set_nsc([3,1,1])
Q=HC.q0
print(Q)

# MFH object of the central region, same kT and U as electrodes!
MFH_HC = HubbardHamiltonian(HC, U=U, kT=0.025,nkpt=[kx,1,1],q=(0.5*Q, 0.5*Q))
MFH_HC.set_polarization([112,120,128,138,147,152], dn=[115,121,129,139,146,155])

# First create the NEGF object, where we pass the MFH converged electrodes and
# the central region HubbardHamiltonian object
elec_indx = [range(len(H_elec)), range(-len(H_elec), 0)]
negf = NEGF(MFH_HC, [(MFH_elec, '-C'), (MFH_elec, '+C')], elec_indx)

# Converge using Green's function method to obtain the densities
MFH_HC.converge(negf.calc_n_open, steps=5, tol=1e-4, print_info=True)

#plot
ds=MFH_HC.n[0]-MFH_HC.n[1]
dE=MFH_HC.H.tocsr(dim=0).diagonal()-MFH_HC.H.tocsr(dim=1).diagonal()

# Create just a figure and only one subplot
fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(10,12),sharex=True)
fig.suptitle(f'Spin polarization',size=24)

pcm1=ax[0].scatter(device[:,2],device[:,0],c=ds,cmap='RdBu',vmin=-0.25,vmax=0.25,edgecolors='k',s=300)
fig.colorbar(pcm1, ax=ax[0],).set_label(label=r'${Q_\uparrow}-{Q_\downarrow}$',size=15)
ax[0].set_ylabel('x Å',size=16)

for i, txt in enumerate(ds):
    if np.abs(txt)>0.05:
        ax[0].annotate(np.round(txt,2), (device.xyz[i,2]-1.8, device.xyz[i,0]),size=10)
        
pcm2 = ax[1].scatter(device[:,2],device[:,0],c=dE,cmap='RdBu_r',vmin=-0.5,vmax=0.5,edgecolors='k',s=300)
fig.colorbar(pcm2, ax=ax[1]).set_label(label=r'${E_\uparrow}-{E_\downarrow}$',size=15)

ax[1].set_ylabel('x Å',size=16)
ax[1].set_xlabel('z Å',size=16)
fig.tight_layout()
fig.savefig(f'{newpath}/SP_Eq.png',dpi=200)
plt.clf()
MFH_HC.write_density('n_EQ.nc')
MFH_HC.H.write('H_EQ.nc')
###################Device Calculation (NEQ)################################
# Build sisl.Hamiltonian object using the sp2 function
HC= sp2(device, t1=2.7, t2=0.0, t3=0.0, spin='polarized',dq=dN)
# with periodic boundary conditions
HC.set_nsc([3,1,1])
# set potential
init_V(HC,V=V,elec_indx=elec_indx,axis=2)

# MFH object of the central region, same kT and U as electrodes!
MFH_HC_neq = HubbardHamiltonian(HC, U=U, kT=0.025,nkpt=[kx,1,1],q=(0.5*Q, 0.5*Q))
# Read spin density and set spin polarization
MFH_HC_neq.set_polarization([112,120,128,138,147,152], dn=[115,121,129,139,146,155])

# First create the NEGF object, where we pass the MFH converged electrodes and
# the central region HubbardHamiltonian object
elec_indx = [range(len(H_elec)), range(-len(H_elec), 0)]
negf = NEGF(MFH_HC_neq, [(MFH_elec, '-C'), (MFH_elec, '+C')], elec_indx,V=V,H_eq=MFH_HC)

# take indexes for ploting changes in charges
a_dev = []
elec_idx = []
for ia in np.ravel(negf.elec_idx):
    if ia < 0:
        elec_idx.append(range(MFH_HC_neq.sites)[ia])
    else:
        elec_idx.append(ia)

for ia in range(MFH_HC_neq.sites):
    if ia not in elec_idx:
        a_dev.append(ia)
        
# iterate scf calculations
for i in np.arange(0,30):
    print(f'Scf iteration:{i}')
    onsite_old=MFH_HC_neq.TBHam.tocsr(dim=0).diagonal()
    #######################################################################################################
    # Create figure 
    fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(10,12),sharex=True)
    fig.suptitle(f'Scf iteration:{i} {dQ} e',size=24)
    pcm1=ax[0].scatter(device[:,2],device[:,0],c=onsite_old,cmap='RdBu_r',vmin=-0.5*V,vmax=0.5*V,edgecolors='k',s=300)
    fig.colorbar(pcm1, ax=ax[0]).set_label(label=r'E (eV)',size=15,)
    ax[0].set_ylabel('x Å',size=16)
    ax[1].plot(device[:,2],onsite_old,c='k')
    pcm2=ax[1].scatter(device[:,2],onsite_old,c=onsite_old,cmap='RdBu_r',vmin=-0.5*V,vmax=0.5*V,edgecolors='k',s=300)
    fig.colorbar(pcm2, ax=ax[1]).set_label(label=r'E (eV)',size=15,)
    ax[1].set_xlabel('z Å',size=16)
    ax[1].set_ylabel(r'E (eV)',size=16)
    #  save figure
    fig.tight_layout()
    fig.savefig(f'{newpath}/fig_{i}.png',dpi=200)
    plt.clf() 
    #######################################################################################################
    # Create just a figure and only one subplot
    ds=MFH_HC_neq.n[0]-MFH_HC_neq.n[1]
    dE=MFH_HC_neq.H.tocsr(dim=0).diagonal()-MFH_HC_neq.H.tocsr(dim=1).diagonal()
    
    fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(10,12),sharex=True)
    fig.suptitle(f'Spin polarization iteration {i} {dQ} e',size=24)
    pcm1=ax[0].scatter(device[:,2],device[:,0],c=ds,cmap='RdBu',vmin=-0.25,vmax=0.25,edgecolors='k',s=300)
    fig.colorbar(pcm1, ax=ax[0],).set_label(label=r'${Q_\uparrow}-{Q_\downarrow}$',size=15)
    ax[0].set_ylabel('x Å',size=16)
    for idx_txt, txt in enumerate(ds):
    	if np.abs(txt)>0.05:
    		ax[0].annotate(np.round(txt,2), (device.xyz[idx_txt,2]-1.8, device.xyz[idx_txt,0]),size=10)
    pcm2 = ax[1].scatter(device[:,2],device[:,0],c=dE,cmap='RdBu_r',vmin=-0.5,vmax=0.5,edgecolors='k',s=300)
    fig.colorbar(pcm2, ax=ax[1]).set_label(label=r'${E_\uparrow}-{E_\downarrow}$',size=15)
    ax[1].set_ylabel('x Å',size=16)
    ax[1].set_xlabel('z Å',size=16)
    fig.tight_layout()
    fig.savefig(f'{newpath}/sp_{i}.png',dpi=200)
    plt.clf()
    #######################################################################################################
    # Scf calculation

    result=MFH_HC_neq.iterate(negf.calc_n_open, **{'qtol':1e-02,'method':met,'lambda_tf':ltf ,'mixing':'damped','C_self':C_self})
    
    #######################################################################################################
    # Create plot of q and dV
    q_neq = MFH_HC_neq.n.sum(axis=0)
    q_eq = MFH_HC.n.sum(axis=0)
    dq = (q_neq - q_eq)[a_dev]

    onsite_new=MFH_HC_neq.TBHam.tocsr(dim=0).diagonal()
    donsite=onsite_new-onsite_old
    
    # Create just a figure and only one subplot
    fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(10,12),sharex=True)
    fig.suptitle(f'Change in charges after iteration {i}',size=24)
    pcm1=ax[0].scatter(device[a_dev][:,2],device[a_dev][:,0],c=dq,cmap='RdBu_r',vmin=-0.1,vmax=0.1,edgecolors='k',s=300)
    fig.colorbar(pcm1, ax=ax[0],).set_label(label=r'${Q_{neq}-Q_{eq}}$',size=15)
    ax[0].set_ylabel('x Å',size=16)
    
    pcm2 = ax[1].scatter(device[:,2],device[:,0],c=donsite,cmap='RdBu_r',vmin=-0.1,vmax=0.1,edgecolors='k',s=300)
    fig.colorbar(pcm2, ax=ax[1]).set_label(label=r'${E_{new}-E_{old}}$',size=15)
    ax[1].set_ylabel('x Å',size=16)
    ax[1].set_xlabel('z Å',size=16)
    fig.tight_layout()
    fig.savefig(f'{newpath}/dq_{i}.png',dpi=200)
    plt.clf()
    print(f'Change of density of {result}')
    
MFH_HC_neq.write_density('n_NEQ.nc')
MFH_HC_neq.H.write('H_NEQ.nc')

    
