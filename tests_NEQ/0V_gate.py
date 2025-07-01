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

# Set Hubbard U 
U=3.5
kx=1

newpath = f'./Eq_gate_{U}U_{kx}kx' 
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

# get unbias hamiltonian
H0 = sp2(elec_ts, t1=2.7, t2=0.0, t3=0.0, spin='polarized')
q0 = H0.q0

for gate in np.round(np.arange(-0.25,0.375,0.125),3):
    #############################################################Electrodes#######################################################################
    # calcullate charge on each atom
	dn=gate/q0
    #assign charge to electrode hamiltonian
	H_elec = sp2(elec_ts, t1=2.7, t2=0.0, t3=0.0, spin='polarized',dq=dn)
    #get charge on the gated electrode
	q=H_elec.q0
    #pass equal amount of charge on MFH object
	MFH_elec = HubbardHamiltonian(H_elec, U=U, nkpt=[30,1,30],q=(0.5*q, 0.5*q), kT=0.025) 
	MFH_elec.random_density()
	MFH_elec.converge(density.calc_n, tol=1e-6, print_info=True, steps=10)
	print('Fermi level of'+str(MFH_elec.fermi_level()))
	MFH_elec.shift(MFH_elec.fermi_level())
	    #############################################################Plot Bands#######################################################################
	fig, ax = plt.subplots(figsize=(2,8))
	fig.suptitle(f'Spin polarization e {gate}',size=24)
	bs = sisl.BandStructure(MFH_elec.H, [[0.5, 0, 0],[0, 0, 0], [0.0, 0, 0.5]], 100,['X', r'Gamma', 'Z'])
	lk = bs.lineark(ticks=False)
	bs_eig = bs.apply.array.eigh()
	plt.plot(lk, bs_eig,c='red')
	plt.ylim([-4,4])
	plt.fill_between(lk, y1=-10, y2=0,color='red',alpha=0.25)
	plt.xlabel('k', size=14)
	plt.ylabel('E (eV)', size=14)
	fig.savefig(f'{newpath}/Electrode_Bandstructure_{gate}.png',dpi=200,bbox_inches="tight")
	plt.clf()
	#############################################################Device#######################################################################
	
    # calculate charge on device on each atom 
	dQ=(gate)*8
	dN=dQ/device.na
	print(dN,dQ)
	HC= sp2(device, t1=2.7, t2=0.0, t3=0.0, spin='polarized',dq=dN)
	# with periodic boundary conditions
	HC.set_nsc([3,1,1])
    # get total charge
	Q=HC.q0
	# MFH object of the central region, same kT and U as electrodes!
	MFH_HC = HubbardHamiltonian(HC, U=U, kT=0.025, nkpt=[kx,1,1],q=(0.5*Q, 0.5*Q))
	MFH_HC.set_polarization([112,120,128,138,147,152], dn=[115,121,129,139,146,155])
	elec_indx = [range(len(H_elec)), range(-len(H_elec), 0)]
	negf = NEGF(MFH_HC, [(MFH_elec, '-C'), (MFH_elec, '+C')], elec_indx)
	MFH_HC.converge(negf.calc_n_open, steps=5, tol=1e-4, print_info=True)
	    #############################################################Plot SP#######################################################################
	ds=MFH_HC.n[0]-MFH_HC.n[1]
	dE=MFH_HC.H.tocsr(dim=0).diagonal()-MFH_HC.H.tocsr(dim=1).diagonal()
	# Create just a figure and only one subplot
	fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(10,12),sharex=True)
	fig.suptitle(f'Spin Polarization e {dQ}',size=24)
	
	#spin polarization
	pcm1=ax[0].scatter(device[:,2],device[:,0],c=ds,cmap='RdBu_r',vmin=-0.25,vmax=0.25,edgecolors='k',s=300)
	fig.colorbar(pcm1, ax=ax[0],).set_label(label=r'${Q_\uparrow}-{Q_\downarrow}$',size=15,weight='bold')
	ax[0].set_ylabel('x Å',size=16)
	for i, txt in enumerate(ds):
		if np.abs(txt)>0.05:
			ax[0].annotate(np.round(txt,2), (device.xyz[i,2]-1.8, device.xyz[i,0]),size=10)
	# onsite difference        
	pcm2 = ax[1].scatter(device[:,2],device[:,0],c=dE,cmap='RdBu_r',vmin=-0.5,vmax=0.5,edgecolors='k',s=300)
	fig.colorbar(pcm2, ax=ax[1]).set_label(label=r'${E_\uparrow}-{E_\downarrow}$',size=15,weight='bold')
	ax[1].set_ylabel('x Å',size=16)
	ax[1].set_xlabel('z Å',size=16)
	fig.tight_layout()
	fig.savefig(f'{newpath}/SP_{dQ}.png',dpi=200,bbox_inches="tight")
	plt.clf()
	
###########################################################Plot DOS######################################################################
	fig, ax = plt.subplots(figsize=(2,8),)
	DOS_0=MFH_HC.DOS(np.arange(-1,1,0.01),spin=0)
	DOS_1=MFH_HC.DOS(np.arange(-1,1,0.01),spin=1)
	plt.plot(DOS_0,np.arange(-1,1,0.01),label='spin0',c='r')
	plt.plot(DOS_1,np.arange(-1,1,0.01),label='spin1',c='b')
	plt.ylabel('E (eV)',size=16)
	plt.xlabel('DOS',size=16)
	fig.savefig(f'{newpath}/DOS_{dQ}.png',dpi=200,bbox_inches="tight")
	plt.clf()
	print("finish calculation")
