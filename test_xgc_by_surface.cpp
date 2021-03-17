#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "adios2.h"
#include "mgard_api.h"

#define SECONDS(d) std::chrono::duration_cast<std::chrono::seconds>(d).count()

template <typename Type>
void FileWriter_bin(const char *filename, Type *data, size_t size)
{
    std::ofstream fout(filename, std::ios::binary);
    fout.write((const char*)(data), size*sizeof(Type));
    fout.close();
}


template<typename Type>
void FileWriter_ad(const char *filename, Type *data, std::vector<size_t> global_dim,
    std::vector<size_t>local_dim, size_t offset, bool rel)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rel) {
        size_t tot_sz = local_dim[0] * local_dim[1] * local_dim[2] * local_dim[3];
        for (size_t it=0; it<tot_sz; it++)
            data[it] = exp(data[it]);
    }
    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO bpIO = ad.DeclareIO("WriteBP_File");
    std::cout << "processor " << rank << ": {" << global_dim[0] << ", " << global_dim[1] << ", " << global_dim[2] << ", " << global_dim[3] << "}, {";
    std::cout << local_dim[0] << ", " <<local_dim[1] << ", " << local_dim[2] << ", " << local_dim[3] << "}, " << offset << "\n";
    adios2::Engine bpFileWriter = bpIO.Open(filename, adios2::Mode::Write);
    if (data!=NULL) {
        if (bpIO.InquireVariable<double>("i_f")) {
            adios2::Variable<Type> bp_fdata = bpIO.InquireVariable<double>("i_f");
            bp_fdata.SetSelection(adios2::Box<adios2::Dims>({0, 0, offset, 0}, local_dim)); 
            bpFileWriter.Put<Type>(bp_fdata, data);
        } else {
            adios2::Variable<Type> bp_fdata = bpIO.DefineVariable<Type>(
                  "i_f", global_dim, {0, 0, offset, 0}, local_dim,  adios2::ConstantDims);
            // Engine derived class, spawned to start IO operations //
            bpFileWriter.Put<Type>(bp_fdata, data);
        }
    }
    bpFileWriter.Close();
    std::cout << "processor " << rank << " finish FileWriter -- " << filename << "\n";
}

template<typename Type>
const double* mgard_reconstruct_3D(Type *data, std::vector<size_t> shape, size_t timeStep)
{
    const std::array<size_t, 3> dims = {shape[0], shape[1], shape[2]};
    const mgard::TensorMeshHierarchy<3, double> hierarchy(dims);
    const size_t ndof = hierarchy.ndof();
    double tol = 1e13;
    const mgard::CompressedDataset<3, double> compressed =
        mgard::compress(hierarchy, data, 0.0, tol);
//    std::cout << "after compression: " << compressed.size() << "\n";
    const mgard::DecompressedDataset<3, Type> decompressed = mgard::decompress(compressed);
//    FileWriter_ad(("i_f.mgard." + std::to_string(timeStep) + ".bp").c_str(), (double *)decompressed.data(), {shape[0], 1, shape[1], shape[2]});
    return decompressed.data();
}


template<typename Type>
const double* mgard_reconstruct_4D(Type *data, std::vector<size_t> shape, size_t timeStep)
{
    const std::array<size_t, 4> dims = {shape[0], shape[1], shape[2], shape[3]}; 
    const mgard::TensorMeshHierarchy<4, double> hierarchy(dims);
    const size_t ndof = hierarchy.ndof();
    double tol = 1e13;
    const mgard::CompressedDataset<4, double> compressed =
        mgard::compress(hierarchy, data, 0.0, tol);
//    std::cout << "after compression: " << compressed.size() << "\n";
    const mgard::DecompressedDataset<4, double> decompressed = mgard::decompress(compressed);
//    FileWriter_ad(("i_f.mgard." + std::to_string(timeStep) + ".bp").c_str(), (double *)decompressed.data(), shape);
    return decompressed.data();
}

template<typename Real>
std::vector<double> L_inif_L2_error(Real *ori, Real *rct, size_t count)
{
    double l2_e = 0.0, l_inf = 0.0, err;
    for (size_t i=0; i<count; i++) {
        err = abs(ori[i] - rct[i]);
        l_inf  = std::max(l_inf, err); 
        l2_e  += err*err; 
    }
    std::vector<double>err_vec{l_inf, l2_e/(double)count};
    return err_vec;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, np_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np_size);

	std::chrono::steady_clock clock;
	std::chrono::steady_clock::time_point start, stop;
	std::chrono::steady_clock::duration duration;

    double tol;
    bool rel = (strcmp(argv[1], "rel") == 0);
    tol = atof(argv[2]);
    if (rel) tol = log(1+tol);
    char datapath[2048], filename[2048], readin_f[2048];
    strcpy(datapath, argv[3]);
    strcpy(filename, argv[4]);
    sprintf(readin_f, "%s%s", datapath, filename);
    strcat(filename, ".mgard");
    size_t timeSteps = atoi(argv[5]);
    int pad_elem = 8, min_elem = 0, max_elem = 0;
    std::vector<int> shape_pscan(timeSteps, 0); 
    if (rank==0) {
        if (rel) std::cout << "relative eb = " << tol << "\n";
        else std::cout << "absolute eb = " << tol << "\n";
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
            max_elem = std::max(max_elem, std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>())); 
            if (var_i_f_in.Shape()[0]>1)
                pad_elem = std::min(pad_elem, (int)shape[0]);
        }
        shape_pscan[timeStep] = (timeStep==0) ? shape[0] : (shape_pscan[timeStep-1]+shape[0]); 
//        std::cout << shape[0]  << ", " <<  shape[1] << ", " << shape[2] << ", " <<  shape[3] << "\n";
        if (timeStep+np_size>timeSteps-1) {
            min_elem = pad_elem * shape[1] * shape[2] * shape[3];
            vx   = shape[2];
            vy   = shape[3];
            nphi = shape[1];
        }
    }
    reader_s0.Close();
    std::cout << "rank: " << rank << ", max_elem: " << max_elem << ", min_elem: " << min_elem << ", pad_elem: " << pad_elem <<  " total sz: " << shape_pscan[timeSteps-1] << " allocated sz: " << max_elem + min_elem*2<<"\n";

    adios2::IO bpIO = ad.DeclareIO("WriteBP_File");
    adios2::Engine bpFileWriter = bpIO.Open(filename, adios2::Mode::Write);
    adios2::Variable<double> bp_fdata = bpIO.DefineVariable<double>("i_f",  {(size_t)shape_pscan[timeSteps-1], nphi, vx, vy});
   
    double *i_f_padding = new double[max_elem + min_elem*2];

    adios2::IO reader_io = ad.DeclareIO("XGC");
    adios2::Engine reader = reader_io.Open(readin_f, adios2::Mode::Read);
    // read data
    size_t rct_sec = 0;
    adios2::fstep iStep;
    double l2_e = 0.0, l_inf=0.0;
    size_t compressed_sz = 0;
    size_t original_sz = 0;
    int timeStep = rank;
    start = clock.now();
    for (timeStep=0; timeStep < timeSteps; timeStep++) { 
        reader.BeginStep();
        if ((timeStep % np_size) != rank) { 
            reader.EndStep();
            continue;
        }
        var_i_f_in = reader_io.InquireVariable<double>("i_f");
        if (!var_i_f_in) {
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
        size_t tot_elem = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        if (shape[0] < 2) { // isolated mesh node, no need for padding
            const std::array<size_t, 3> dims = {shape[1], shape[2], shape[3]};
            const mgard::TensorMeshHierarchy<3, double> hierarchy(dims);
            std::cout << shape[0] << ", " << shape[2] << ", " << shape[3] << " -- "<< tol << "\n";
            const mgard::CompressedDataset<3, double> compressed = mgard::compress(hierarchy, i_f_in.data(), 0.0, tol);
            compressed_sz += compressed.size();
            const mgard::DecompressedDataset<3, double> decompressed = mgard::decompress(compressed);
            // ADIOS write
            bp_fdata.SetSelection(adios2::Box<adios2::Dims>({offset_d1, 0, 0, 0}, shape));
            bpFileWriter.Put<double>(bp_fdata, (double *)decompressed.data());
            bpFileWriter.PerformPuts();
            // Engine derived class, spawned to start IO operations //
        } else {
            // padding using the periodic boundary condition
            memcpy(&i_f_padding[min_elem], i_f_in.data(), tot_elem*sizeof(double));         
            size_t step = shape[1]*shape[2]*shape[3];
            for (size_t ipd=0; ipd<pad_elem; ipd++) {
                size_t k = ipd*step;
                memcpy(&i_f_padding[k], &i_f_in[tot_elem-k-step], step*sizeof(double));
                memcpy(&i_f_padding[min_elem+tot_elem+k], &i_f_in[k], step*sizeof(double));
            }
            // MGARD compression
            const std::array<size_t, 4> dims = {shape[0]+pad_elem*2, shape[1], shape[2], shape[3]};
            std::cout << shape[0]+pad_elem*2 << ", " << shape[1] << ", " << shape[2] << ", " << shape[3] << "\n";
            const mgard::TensorMeshHierarchy<4, double> hierarchy(dims);
            const mgard::CompressedDataset<4, double> compressed = mgard::compress(hierarchy, i_f_padding, 0.0, tol);
            compressed_sz += compressed.size();
            const mgard::DecompressedDataset<4, double> decompressed = mgard::decompress(compressed);
            // ADIOS write
            bp_fdata.SetSelection(adios2::Box<adios2::Dims>({offset_d1, 0, 0, 0}, shape));
            bpFileWriter.Put<double>(bp_fdata, ((double *)decompressed.data())+min_elem);
            bpFileWriter.PerformPuts();
        } 
        original_sz   += tot_elem;
        stop = clock.now();
        rct_sec += SECONDS(stop - start); 
    } 
    reader.Close();
    bpFileWriter.Close();
    delete i_f_padding;
//    std::cout << "Total reconstruction cost: " << std::floor(rct_sec/60.0) << " min and " << rct_sec%60 << " sec\n";
    std::cout << "compression ratio: " << ((double)original_sz*8/compressed_sz) << ", ori: " << original_sz << ", compressed: " << compressed_sz << "\n";
    MPI_Finalize();
}
