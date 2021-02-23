import numpy as np
import adios2 as ad2
from tqdm import tqdm

import os
import subprocess

class XGC:

    class Mesh:
        def __init__(self, expdir=''):
            fname = os.path.join(expdir, 'xgc.mesh.bp')
            print (f"Reading: {fname}")
            with ad2.open(fname, 'r') as f:
                self.nnodes = f.read('n_n').item()
                self.ncells = f.read('n_t').item()
                self.rz = f.read('rz')
                self.conn = f.read('nd_connect_list')
                self.psi = f.read('psi')

            self.r = self.rz[:,0]
            self.z = self.rz[:,1]

    class F0mesh:
        def __init__(self, expdir=''):
            fname = os.path.join(expdir, 'xgc.f0.mesh.bp')
            print (f"Reading: {fname}")
            with ad2.open(fname, 'r') as f:
                self.f0_nmu = f.read('f0_nmu')
                self.f0_nvp = f.read('f0_nvp')
                self.f0_smu_max = f.read('f0_smu_max')
                self.f0_dsmu = f.read('f0_dsmu')
                self.f0_dvp = f.read('f0_dvp')
                self.f0_T_ev = f.read('f0_T_ev')
                self.f0_grid_vol_vonly = f.read('f0_grid_vol_vonly')

    class Grid:
        class Mat:
            """
             integer :: n,m,width
             real (8), allocatable :: value(:,:)
             integer, allocatable :: eindex(:,:),nelement(:)
            """
            n,m,width = 0,0,0
            value = None
            eindex = None
            nelement = None

            def mat_transpose_mult(self, x):
                assert self.n == len(x)
                y = np.zeros([self.m,])
                for i in range(self.n):
                    for j in range(self.nelement[i]):
                        k = self.eindex[i,j]-1
                        y[k] = y[k] + self.value[i,j] * x[i]
                return y

        def __init__(self, expdir=''):
            self.cnv_2d_00 = self.Mat()
            
            fname = os.path.join(expdir, 'xgc.fluxavg.bp')
            print (f"Reading: {fname}")
            with ad2.open(fname, 'r') as f:
                self.cnv_2d_00.n = f.read('nnode')
                self.cnv_2d_00.m = f.read('npsi')
                self.cnv_2d_00.width = f.read('width')
                self.cnv_2d_00.value = f.read('value')
                self.cnv_2d_00.eindex = f.read('eindex')
                self.cnv_2d_00.nelement = f.read('nelement')
                self.cnv_norm_1d00 = f.read('norm1d')
                self.cnv_norm_2d = f.read('norm2d')
                self.npsi = f.read('npsi').item()
    
            fname = os.path.join(expdir, 'xgc.mesh.bp')
            print (f"Reading: {fname}")
            with ad2.open(fname, 'r') as f:
                self.psi = f.read('psi')
                self.nnodes = f.read('n_n')
                self.x = f.read('rz')

    """
      ! check if region 1 or 2
      logical function is_rgn12(r,z,psi)
        implicit none
        real (8) :: r,z,psi

        ! Use better logic for double-null configurations
        !if((psi > eq_x_psi -epsil_psi .or. &amp;
        !     -(r-eq_x_r)*eq_x_slope + (z-eq_x_z) > 0D0) .and. -(r-eq_x2_r)*eq_x2_slope + (z-eq_x2_z) < 0D0   ) then
        !   is_rgn12=.true.
        !else
        !   is_rgn12=.false.
        !endif
        if ( (psi .le. eq_x_psi-epsil_psi .and. -(r-eq_x_r)*eq_x_slope + (z-eq_x_z) > 0D0 .and. &amp;
              -(r-eq_x2_r)*eq_x2_slope + (z-eq_x2_z) < 0D0) .or. &amp;
             (psi .gt. eq_x_psi-epsil_psi .and. psi .le. eq_x2_psi-epsil_psi .and. &amp;
              -(r-eq_x2_r)*eq_x2_slope + (z-eq_x2_z) < 0D0) .or. &amp;
             psi .gt. eq_x2_psi-epsil_psi) then
           is_rgn12=.true.
        else
           is_rgn12=.false.
        endif
      end function is_rgn12
    """
    def is_rgn12(self, r,z,psi):
        if ((psi<= self.eq_x_psi-self.epsil_psi and -(r-self.eq_x_r)*self.eq_x_slope + (z-self.eq_x_z) > 0.0 and \
              -(r-self.eq_x2_r)*self.eq_x2_slope + (z-self.eq_x2_z) < 0.0) or \
            (psi > self.eq_x_psi-self.epsil_psi and psi <= self.eq_x2_psi-self.epsil_psi and \
             -(r-self.eq_x2_r)*self.eq_x2_slope + (z-self.eq_x2_z) < 0.0) or \
            psi > self.eq_x2_psi-self.epsil_psi):
            return True
        else:
            return False

    """
    subroutine convert_grid_2_001d(grid,v2d,v1d)
      use grid_class
      implicit none
      type(grid_type), intent(in) :: grid
      real (8), intent(in) :: v2d(grid%nnode)
    #ifdef CONVERT_GRID2
      real (8), intent(out)  :: v1d(grid%npsi_surf)
    #else
      real (8), intent(out)  :: v1d(grid%npsi00)
    #endif

      call mat_transpose_mult(grid%cnv_2d_00,v2d,v1d)
      v1d=v1d/grid%cnv_norm_1d00

    end subroutine convert_grid_2_001d
    """
    def convert_grid_2_001d(self, v2d):
        v1d = self.grid.cnv_2d_00.mat_transpose_mult(v2d)
        v1d = v1d/self.grid.cnv_norm_1d00
        return v1d

    """
    subroutine convert_001d_2_grid(grid,v1d,v2d)
      use eq_module
      use grid_class
      implicit none
      type(grid_type), intent(in) :: grid
      real (8), intent(in)  :: v1d(grid%npsi00)
      real (8), intent(out) :: v2d(grid%nnode)
      !
      integer :: i, ip
      real (8) :: pn, wp

      do i=1, grid%nnode
         pn=(grid%psi(i)-grid%psi00min)/grid%dpsi00
         ip=floor(pn)+1
         if(0 < ip .and. ip < grid%npsi00 .and. is_rgn12(grid%x(1,i),grid%x(2,i),grid%psi(i)) ) then
            wp=1D0 - ( pn - real(ip-1,8) )
         elseif( ip <= 0 ) then
            ip=1
            wp=1D0
         else
            ip=grid%npsi00-1
            wp=0D0
         endif

         v2d(i)=v1d(ip)*wp  + v1d(ip+1)*(1D0-wp)
      end do

    end subroutine convert_001d_2_grid
    """
    def convert_001d_2_grid(self, v1d):
        v2d = np.zeros(self.grid.nnodes)
        for i in range(self.grid.nnodes):
            pn=(self.grid.psi[i]-self.grid.psi00min)/self.grid.dpsi00
            ip=int(np.floor(pn))+1
            if(0 < ip and ip < self.grid.npsi00 and self.is_rgn12(self.grid.x[i,0],self.grid.x[i,1],self.grid.psi[i])):
                wp=1.0 - ( pn - float(ip-1) )
            elif (ip<=0):
                ip=1
                wp=1.0
            else:
                ip=self.grid.npsi00-1
                wp=0.0

            v2d[i]=v1d[ip-1]*wp  + v1d[ip]*(1.0-wp)

        return v2d

    def __init__(self, expdir='/Users/qg7/Documents/Projs/XGC Fusion/data files/low_res'):
        self.expdir = expdir
        
        ## populate mesh
        self.mesh = self.Mesh(expdir)

        ## populate f0mesh
        self.f0mesh = self.F0mesh(expdir)
        
        ## populate grid
        self.grid = self.Grid(expdir)

        ## XGC has been updated to output more values: eq_x_slope, eq_x2_psi, eq_x2_r, eq_x2_z, eq_x2_slope
        fname = os.path.join(expdir, 'xgc.equil.bp')
        print (f"Reading: {fname}")
        with ad2.open(fname, 'r') as f:
            self.eq_axis_b = f.read('eq_axis_b')
            self.eq_axis_r = f.read('eq_axis_r')
            self.eq_axis_z = f.read('eq_axis_z')

            self.eq_max_r = f.read('eq_max_r')
            self.eq_max_z = f.read('eq_max_z')
            self.eq_min_r = f.read('eq_min_r')
            self.eq_min_z = f.read('eq_min_z')

            self.eq_x_psi = f.read('eq_x_psi')
            self.eq_x_r = f.read('eq_x_r')
            self.eq_x_z = f.read('eq_x_z')
            self.eq_x_slope = f.read('eq_x_slope')


            self.eq_x2_psi = f.read('eq_x2_psi')
            self.eq_x2_r = f.read('eq_x2_r')
            self.eq_x2_z = f.read('eq_x2_z')
            self.eq_x2_slope = f.read('eq_x2_slope')

        self.epsil_psi =  1E-5

        fname = os.path.join(expdir, 'fort.input.used')
        result = subprocess.run(['/usr/bin/grep', '-a', 'SML_OUTPSI', fname], stdout=subprocess.PIPE)
        self.sml_00_npsi = self.grid.npsi
        self.sml_inpsi = 0.0000000000000000
        kv = result.stdout.decode().replace(' ','').replace(',','').split('\n')[0].split('=')
        print("kv: ", kv)
        self.sml_outpsi = float(kv[1])
        print ('sml_00_npsi, sml_inpsi, sml_outpsi=', self.sml_00_npsi, self.sml_inpsi, self.sml_outpsi)

        fname = os.path.join(expdir, 'fort.input.used')
        result = subprocess.run(['/usr/bin/grep', '-a', 'SML_NPHI_TOTAL', fname], stdout=subprocess.PIPE)
        kv = result.stdout.decode().replace(' ','').replace(',','').split('\n')[0].split('=')
        self.nphi = int(int(kv[1]))
        print ('sml_nphi_total=', self.nphi)
        
        self.grid.npsi00 = self.sml_00_npsi
        self.grid.psi00min = self.sml_inpsi * self.eq_x_psi
        self.grid.psi00max = self.sml_outpsi * self.eq_x_psi
        self.grid.dpsi00 = (self.grid.psi00max - self.grid.psi00min)/float(self.grid.npsi00-1)

    def f0_diag(self, f0_inode1, ndata, isp, f0_f, progress=False):
        """ 
        Input:
        f0_inode1: int
        ndata: int (f0_inode2=f0_inode1+ndata)
        isp: electron(=0) or ion(=1)
        f0mesh: (F0mesh object) shoul have the following attrs:
            f0_nmu: int
            f0_nvp: int
            f0_smu_max: float
            f0_dsmu: float
            f0_T_ev: (nsp, nnodes)
            f0_grid_vol_vonly: (nsp, nnodes)
            f0_dvp: double
        mesh: (Mesh object) shoul have the following attrs:
            nnodes: int  -- number of mesh nodes
        f0_f: (ndata, f0_nmu+1, 2*f0_nvp+1) -- f-data

        Output: 
        den: (ndata, f0_nmu+1, 2*f0_nvp+1)
        u_para: (ndata, f0_nmu+1, 2*f0_nvp+1)
        T_perp: (ndata, f0_nmu+1, 2*f0_nvp+1)
        T_para: (ndata, f0_nmu+1, 2*f0_nvp+1)
        n0: (ndata)
        T0: (ndata)

        All outputs are before performing flux-surface averaging
        """

        ## Aliases
        f0_nmu = self.f0mesh.f0_nmu
        f0_nvp = self.f0mesh.f0_nvp
        f0_smu_max = self.f0mesh.f0_smu_max
        f0_dsmu = self.f0mesh.f0_dsmu
        f0_T_ev = self.f0mesh.f0_T_ev
        f0_grid_vol_vonly = self.f0mesh.f0_grid_vol_vonly
        f0_dvp = self.f0mesh.f0_dvp    
        nnodes = self.mesh.nnodes

        ## Check
        if f0_f.ndim == 2:
            f0_f = f0_f[np.newaxis,:]
        #print (f0_f.shape, (ndata, f0_nmu+1, f0_nvp*2+1))
        assert(f0_f.shape[0] == ndata)
        assert(f0_f.shape[1] == f0_nmu+1)
        assert(f0_f.shape[2] >= f0_nvp*2+1)

        sml_e_charge=1.6022E-19  ## electron charge (MKS)
        sml_ev2j=sml_e_charge

        ptl_e_mass_au=2E-2
        ptl_mass_au=2E0
        sml_prot_mass=1.6720E-27 ## proton mass (MKS)
        ptl_mass = [ptl_e_mass_au*sml_prot_mass, ptl_mass_au*sml_prot_mass]

        ## index: imu, range: [0, f0_nmu]
        mu_vol = np.ones(f0_nmu+1)
        mu_vol[0] = 0.5
        mu_vol[-1] = 0.5

        ## index: ivp, range: [-f0_nvp, f0_nvp]
        vp_vol = np.ones(f0_nvp*2+1)
        vp_vol[0] = 0.5
        vp_vol[-1] = 0.5

        #f0_smu_max = 3.0
        #f0_dsmu = f0_smu_max/f0_nmu
        mu = (np.arange(f0_nmu+1, dtype=np.float64)*f0_dsmu)**2

        # out
        den = np.zeros((ndata, f0_nmu+1, 2*f0_nvp+1))
        u_para = np.zeros((ndata, f0_nmu+1, 2*f0_nvp+1))
        T_perp = np.zeros((ndata, f0_nmu+1, 2*f0_nvp+1))
        T_para = np.zeros((ndata, f0_nmu+1, 2*f0_nvp+1))

        # 1) Density, parallel flow, and T_perp moments
        for inode in tqdm(range(0, ndata), disable=not progress):
            ## Mesh properties
            en_th = f0_T_ev[isp,f0_inode1+inode]*sml_ev2j # constant
            vth = np.sqrt(en_th/ptl_mass[isp]) # constant
            f0_grid_vol = f0_grid_vol_vonly[isp,f0_inode1+inode] # constant

            for imu in range(0, f0_nmu+1): # range of bins in para 
                for ivp in range(0, f0_nvp*2+1): # range of bins in perp
                    ## Vspace properties
                    vol = f0_grid_vol * mu_vol[imu] * vp_vol[ivp]  
                    vp = (ivp - f0_nvp) * f0_dvp # middle at 0
                    en = 0.5 * mu[imu]

                    f = f0_f[inode, imu, ivp] #f0_f(ivp,inode,imu,isp)
                    den[inode, imu, ivp] = f * vol 
                    u_para[inode, imu, ivp] = f * vol * vp * vth
                    T_perp[inode, imu, ivp] = f * vol * en * vth**2 * ptl_mass[isp]
                    #if (inode==0): print ('imu,inode,ivp,ptl_mass,vth,en,vol=',imu,inode,ivp,ptl_mass[isp],vth,en,vol)

        for inode in range(0, ndata):
            u_para[inode,:] = u_para[inode,:]/np.sum(den[inode,:])
            T_perp[inode,:] = T_perp[inode,:]/np.sum(den[inode,:])/sml_e_charge

        upar = np.sum(u_para, axis=(1,2))

        # 2) T_para moment
        for inode in tqdm(range(0, ndata), disable=not progress):
            ## Mesh properties
            en_th = f0_T_ev[isp,f0_inode1+inode]*sml_ev2j
            vth = np.sqrt(en_th/ptl_mass[isp])
            f0_grid_vol = f0_grid_vol_vonly[isp,f0_inode1+inode]

            for imu in range(0, f0_nmu+1):
                for ivp in range(0, f0_nvp*2+1):
                    ## Vspace properties
                    vol = f0_grid_vol * mu_vol[imu] * vp_vol[ivp]
                    vp = (ivp - f0_nvp) * f0_dvp
                    en = 0.5 * (vp - upar[inode] / vth)**2

                    f = f0_f[inode, imu, ivp] #f0_f(ivp,inode,imu,isp)
                    T_para[inode, imu, ivp] = f * vol * en * vth**2 * ptl_mass[isp]

        for inode in range(0, ndata):
            T_para[inode,:] = 2.0*T_para[inode,:]/np.sum(den[inode,:])/sml_e_charge

        n0 = np.sum(den, axis=(1,2))
        T0 = (2.0*np.sum(T_perp, axis=(1,2))+np.sum(T_para, axis=(1,2)))/3.0

        # 3) Get the flux-surface average of n and T
        #    And the toroidal averages of n, T, and u_par
        # jyc: We need all plane data to get flux-surface average. Call f0_avg_diag

        return (den, u_para, T_perp, T_para, n0, T0)
    
    def f0_avg_diag(self, f0_inode1, ndata, n0_all, T0_all):
        """ 
        Input:
        nphi: int -- total number of planes
        f0_inode1: int -- the starting index of mesh node
        ndata: int (f0_inode2=f0_inode1+ndata) -- the number of mesh node
        nnodes: int  -- number of mesh nodes
        n0_all: (nphi, ndata)
        T0_all: (nphi, ndata)

        Output: 
        n0_avg: (ndata) -- i.e., averaged over planes
        T0_avg: (ndata) -- i.e., averaged over planes

        All outputs are before performing flux-surface averaging
        """
        assert n0_all.shape == (self.nphi, ndata)
        assert T0_all.shape == (self.nphi, ndata)

        ## First, we calcuate average over planes
        n0 = np.sum(n0_all, axis=0)/self.nphi
        T0 = np.sum(T0_all, axis=0)/self.nphi

        ## n0
        n0_avg = np.zeros([self.grid.nnodes,])
        n0_avg[:] = np.nan
        n0_avg[f0_inode1:f0_inode1+ndata] = n0

        tmp00_surf = self.convert_grid_2_001d(n0_avg)
        n0_avg = self.convert_001d_2_grid(tmp00_surf)
        # jyc: we disable for debug
        #n0_avg[np.logical_or(np.isinf(n0_avg), np.isnan(n0_avg), n0_avg < 0.0)] = 1E17

        ## T0
        T0_avg = np.zeros([self.grid.nnodes,])
        T0_avg[:] = np.nan
        T0_avg[f0_inode1:f0_inode1+ndata] = T0

        tmp00_surf = self.convert_grid_2_001d(T0_avg)
        T0_avg = self.convert_001d_2_grid(tmp00_surf)
        # jyc: we disable for debug
        #T0_avg[np.logical_or(np.isinf(T0_avg), np.isnan(T0_avg), T0_avg < 0.0)] = 10E0

        return (n0_avg, T0_avg)

if __name__ == "__main__":
    import argparse
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--expdir', help='exp directory (default: %(default)s)', default='')
    parser.add_argument('--timestep', help='timestep', type=int, default=0)
    parser.add_argument('--ndata', help='ndata', type=int)
    args = parser.parse_args()
    
    xgcexp = XGC(args.expdir)

    fname = os.path.join(args.expdir, 'restart_dir/xgc.f0.%05d.bp'%args.timestep)
    with ad2.open(fname, 'r') as f:
        i_f = f.read('i_f')
    
    nphi = i_f.shape[0]
    iphi = 0
    f0_inode1 = 0
    ndata = i_f.shape[2] if args.ndata is None else args.ndata
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ("device:", device)

    fn0_all = np.zeros([nphi,ndata])
    fT0_all = np.zeros([nphi,ndata])
    for iphi in range(nphi):
        f0_f = np.moveaxis(i_f[iphi,:],1,0)
        f0_f = f0_f[f0_inode1:f0_inode1+ndata,:,:]
        den, upara, Tperp, Tpara, fn0, fT0 = \
            xgcexp.f0_diag(f0_inode1=f0_inode1, ndata=ndata, isp=1, f0_f=f0_f)
        fn0_all[iphi,:] = fn0
        fT0_all[iphi,:] = fT0
    print (den.shape, upara.shape, Tperp.shape, Tpara.shape, fn0.shape, fT0.shape)
    
    fn0_avg, fT0_avg = xgcexp.f0_avg_diag(f0_inode1, ndata, fn0_all, fT0_all)
    print (fn0_avg.shape, fT0_avg.shape)
