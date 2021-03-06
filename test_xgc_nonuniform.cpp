#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iterator>

#include "adios2.h"
#include "mgard/mgard_api.h"

#define SECONDS(d) std::chrono::duration_cast<std::chrono::seconds>(d).count()


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, np_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np_size);

	std::chrono::steady_clock clock;
	std::chrono::steady_clock::time_point start, stop;
	std::chrono::steady_clock::duration duration;

    double rel_tol, tol;
    bool rel = (strcmp(argv[1], "rel") == 0);
    rel_tol = atof(argv[2]);
    if (rel) tol = log(1+rel_tol);
    char datapath[2048], filename[2048], readin_f[2048];
    strcpy(datapath, argv[3]);
    strcpy(filename, argv[4]);
    sprintf(readin_f, "%s%s", datapath, filename);
    strcat(filename, ".mgard.non.fls");
    size_t timeSteps = atoi(argv[5]);
    int pad_elem = 8, min_elem = 0, max_elem = 0;
    std::vector<int> shape_pscan(timeSteps, 0); 
    if (rank==0) {
        std::cout << "relative eb = " << rel_tol << "\n";
        std::cout << "number of timeSteps: " << timeSteps << "\n";
        std::cout << "readin: " << readin_f << "\n";
    }
    adios2::ADIOS ad(MPI_COMM_WORLD);
    // Reader I/0
    adios2::IO reader_io_s0 = ad.DeclareIO("XGC_S0");
    adios2::Engine reader_s0 = reader_io_s0.Open(readin_f, adios2::Mode::Read); 
    adios2::Variable<double> var_i_f_in;
    size_t vx, vy, nphi;
    for (unsigned int timeStep = 0; timeStep < timeSteps; ++timeStep) {
        reader_s0.BeginStep();
        var_i_f_in = reader_io_s0.InquireVariable<double>("i_f");
        std::vector<std::size_t> shape = var_i_f_in.Shape();
        reader_s0.EndStep();
        if ((timeStep % np_size) == rank) {
            max_elem = std::max(max_elem, std::accumulate(shape.begin()+1, shape.end(), 1, std::multiplies<size_t>())); 
            if (var_i_f_in.Shape()[1]>1)
                pad_elem = std::min(pad_elem, (int)shape[1]);
        }
        shape_pscan[timeStep] = (timeStep==0) ? shape[1] : (shape_pscan[timeStep-1]+shape[1]); 
//        std::cout << shape[0]  << ", " <<  shape[1] << ", " << shape[2] << ", " <<  shape[3] << ", " << shape_pscan[timeStep] << "\n";
        if (timeStep+np_size>timeSteps-1) {
            min_elem = pad_elem * shape[2] * shape[3];
            vx   = shape[2];
            vy   = shape[3];
            nphi = shape[0];
        }
    }
    reader_s0.Close();
    std::cout << "rank: " << rank << ", max_elem: " << max_elem << ", min_elem: " << min_elem << ", pad_elem: " << pad_elem <<  " total sz: " << shape_pscan[timeSteps-1] <<"\n";

    nphi = 1;
    adios2::IO bpIO = ad.DeclareIO("WriteBP_File");
    adios2::Engine bpFileWriter = bpIO.Open(filename, adios2::Mode::Write);
    
    double *i_f_padding = new double[max_elem + min_elem*2];
    // read xgc data 
    adios2::IO reader_io = ad.DeclareIO("XGC");
    adios2::Variable<double> bp_fdata = bpIO.DefineVariable<double>("i_f",  {nphi, (size_t)shape_pscan[timeSteps-1], vx, vy});
    adios2::Engine reader = reader_io.Open(readin_f, adios2::Mode::Read);
    // read distance data
    adios2::IO reader_io_dist = ad.DeclareIO("XGC_distance");
    sprintf(readin_f, "%s%s", datapath, "grid_vol.bp");
    adios2::Engine reader_dist = reader_io_dist.Open(readin_f, adios2::Mode::Read);
    adios2::Variable<double> var_dist;

    double base_x = (1.0-1.0/vx)/(vx-1), base_y=(1.0-1.0/vy)/(vy-1), base_t=1.0/timeSteps;
    std::vector<double>coords_x(vx, 0.0), coords_y(vy, 0.0), coords_t(timeSteps, 0.0);
    coords_x.at(0) = 0.5/vx;
    coords_y.at(0) = 0.5/vy;
    coords_t.at(0) = 0.5/timeSteps;
    for (size_t idx=1; idx<vx; idx++) coords_x.at(idx) = base_x * idx + coords_x.at(0);
    for (size_t idx=1; idx<vy; idx++) coords_y.at(idx) = base_y * idx + coords_y.at(0);
    for (size_t idx=1; idx<timeSteps; idx++) coords_t.at(idx) = base_t * idx + coords_t.at(0);

    size_t rct_sec = 0;
    adios2::fstep iStep;
    double l2_e = 0.0, l_inf=0.0;
    size_t compressed_sz = 0;
    size_t original_sz = 0;
    int timeStep = rank;
    for (timeStep=0; timeStep < timeSteps; timeStep++) { 
        reader.BeginStep();
        reader_dist.BeginStep();
        if ((timeStep % np_size) != rank) { 
            reader.EndStep();
            reader_dist.EndStep();
            continue;
        }
        var_i_f_in = reader_io.InquireVariable<double>("i_f");
        var_dist   = reader_io_dist.InquireVariable<double>("f0_grid_vol_vonly");
        if ((!var_i_f_in) || (!var_dist)) {
            std::cout << "wrong variable inquiry...exit\n";
            exit(1);
        }
        std::cout << "step " << timeStep << ": var{" << var_i_f_in.Shape()[0] << ", " << var_i_f_in.Shape()[1] << ", " << var_i_f_in.Shape()[2] << ", " << var_i_f_in.Shape()[3] << "}\n";
        std::vector<std::size_t> shape = var_i_f_in.Shape();
        var_i_f_in.SetSelection(adios2::Box<adios2::Dims>({0, 0, 0, 0}, shape));
        std::vector<double>i_f_in;
        reader.Get<double>(var_i_f_in, i_f_in); 
        reader.EndStep();
        size_t offset_d1= (timeStep==0) ? 0 : shape_pscan[timeStep-1];
        start = clock.now();
        size_t step  = shape[1]*shape[2]*shape[3];
        original_sz += shape[0]*shape[1]*shape[2]*shape[3];
        // distance between nodes at the same flux surface
        std::vector<double>distance_z;
        var_dist.SetSelection(adios2::Box<adios2::Dims>({0}, {shape[1]})); 
        reader_dist.Get<double>(var_dist, distance_z);
        reader_dist.EndStep();
        std::vector<double>coords_z(shape[1]+2*pad_elem);
        std::vector<double>vol_z(shape[1]+2*pad_elem);
        std::copy(distance_z.begin(), distance_z.end(), vol_z.begin()+pad_elem);
        std::copy(distance_z.begin(), distance_z.begin()+pad_elem, vol_z.begin()+shape[1]+pad_elem);
        std::copy(distance_z.end()-pad_elem, distance_z.end(), vol_z.begin());
        double FSL = std::accumulate(vol_z.begin(), vol_z.end(), 0.0);
        double div = 1.0/FSL;
        coords_z.at(0) = vol_z.at(0)*0.5*div;
        for (size_t idx=1; idx<coords_z.size(); idx++) {
            coords_z.at(idx) = (vol_z.at(idx-1) + vol_z.at(idx))*0.5 * div + coords_z.at(idx-1);
        }
        std::cout << "max x: " << coords_x.at(shape[2]-1) << ", max y: " << coords_y.at(shape[3]-1) << ", max z: " << coords_z.at(shape[1]-1) << "\n";
        double max_vol = *std::max_element(vol_z.begin(), vol_z.end());

        if (!rel) {
//            std::vector<double>::iterator max_v = std::max_element(i_f_in.begin(), i_f_in.end());
            tol = rel_tol * 7.97125/max_vol;//(*max_v) * rel_tol;
        }
        std::cout << "rank " << rank << ", step: " << timeStep << ", tol: " << tol << "\n";
        if (shape[1] < 2) { // isolated mesh node, no need for padding
            const std::array<size_t, 2> dims = {shape[2], shape[3]};
            const mgard::TensorMeshHierarchy<2, double> hierarchy(dims);
            for (size_t iphi=0; iphi<nphi; iphi++) {
                const mgard::CompressedDataset<2, double> compressed = mgard::compress(hierarchy, (i_f_in.data()+step*iphi), 0.0, tol);
                compressed_sz += compressed.size();
                const mgard::DecompressedDataset<2, double> decompressed = mgard::decompress(compressed);
                // ADIOS write
                bp_fdata.SetSelection(adios2::Box<adios2::Dims>({iphi, offset_d1, 0, 0}, {1, shape[1], vx, vy}));
                bpFileWriter.Put<double>(bp_fdata, (double *)decompressed.data());
                bpFileWriter.PerformPuts();
            // Engine derived class, spawned to start IO operations //
            } 
        } else {
            // non-uniform spacing
            const std::array<std::vector<double>, 3> coords = {coords_z, coords_y, coords_x};
            const std::array<size_t, 3> dims = {shape[1]+pad_elem*2, shape[2], shape[3]};
//            std::cout << shape[1]+pad_elem*2 << ", " << shape[2] << ", " << shape[3] << ", " << min_elem << "\n";
            const mgard::TensorMeshHierarchy<3, double> hierarchy(dims, coords);
            // padding using the periodic boundary condition
            for (size_t iphi=0; iphi<nphi; iphi++) {
                double *data_pos = i_f_in.data() + step*iphi; 
                memcpy(&i_f_padding[min_elem], data_pos, step*sizeof(double));         
                size_t s_padding = shape[2]*shape[3];
                for (size_t ipd=0; ipd<pad_elem; ipd++) {
                    size_t k = ipd*s_padding;
                    memcpy(&i_f_padding[k], &data_pos[step-(pad_elem-ipd)*s_padding], s_padding*sizeof(double));
                    memcpy(&i_f_padding[min_elem+step+k], &data_pos[k], s_padding*sizeof(double));
                }
                // MGARD compression
                const mgard::CompressedDataset<3, double> compressed = mgard::compress(hierarchy, i_f_padding, 0.0, tol);
                compressed_sz += compressed.size();
                const mgard::DecompressedDataset<3, double> decompressed = mgard::decompress(compressed);
                // ADIOS write
                bp_fdata.SetSelection(adios2::Box<adios2::Dims>({iphi, offset_d1, 0, 0}, {1, shape[1], vx, vy}));
//                std::cout << iphi << "," << offset_d1 << "," << vx << "," << vy << ","<< shape[1]<<"\n"; 
                bpFileWriter.Put<double>(bp_fdata, ((double *)decompressed.data())+min_elem);
                bpFileWriter.PerformPuts();
            }
        } 
        stop = clock.now();
        rct_sec += SECONDS(stop - start); 
    } 
    reader.Close();
    reader_dist.Close();
    bpFileWriter.Close();
    delete i_f_padding;
//    std::cout << "Total reconstruction cost: " << std::floor(rct_sec/60.0) << " min and " << rct_sec%60 << " sec\n";
    std::cout << "processor " << rank  << " original_size = " << original_sz << ", after compression = " << compressed_sz << "\n"; 
    std::cout << "compression ratio: " << ((double)original_sz*8.0/compressed_sz) << "\n"; 
    MPI_Finalize();
}
