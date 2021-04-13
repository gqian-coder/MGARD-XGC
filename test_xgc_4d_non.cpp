#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

#include "adios2.h"
#include "mgard/mgard_api.h"

// only compress the plane 0 

#define SECONDS(d) std::chrono::duration_cast<std::chrono::seconds>(d).count()

template <typename Type>
void FileWriter_bin(const char *filename, Type *data, size_t size)
{
  std::ofstream fout(filename, std::ios::binary);
  fout.write((const char*)(data), size*sizeof(Type));
  fout.close();
}

template<typename Type>
void FileWriter_ad(const char *filename, Type *data, std::vector<size_t> global_dim, bool rel)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rel) {
        size_t tot_sz = std::accumulate(begin(global_dim), end(global_dim), 1, std::multiplies<double>()); 
        for (size_t it=0; it<tot_sz; it++)
            data[it] = exp(data[it]);
    }
    adios2::ADIOS ad;
    adios2::IO bpIO = ad.DeclareIO("WriteBP_File");
    std::cout << "processor " << rank << ": {" << global_dim[0] << ", " << global_dim[1] << ", " << global_dim[2] << ", " << global_dim[3] << "}\n";
    adios2::Variable<Type> bp_fdata = bpIO.DefineVariable<Type>(
          "i_f_4d", global_dim, {0, 0, 0, 0}, global_dim,  adios2::ConstantDims);
    // Engine derived class, spawned to start IO operations //
    printf("write...%s\n", filename);
    adios2::Engine bpFileWriter = bpIO.Open(filename, adios2::Mode::Write);
    bpFileWriter.Put<Type>(bp_fdata, data);
    bpFileWriter.Close();
    std::cout << "processor " << rank << " finish FileWriter_ad\n";
}

template<typename Real>
std::vector<double> L_inif_L2_error(Real *ori, Real *rct, size_t count)
{
    double l2_e = 0.0, l_inf = 0.0, err;
    std::cout << "count in L-err: " << count << "\n";
    for (size_t i=0; i<count; i++) {
        err = abs(ori[i] - rct[i]);
        l_inf  = std::max(l_inf, err);
        l2_e  += err*err;
    }
    std::cout << "l-inf: " << l_inf << ", " << "l2-norm: " << sqrt(l2_e/(double)count) << "\n";
    std::vector<double>err_vec{l_inf, l2_e/(double)count};
    return err_vec;
}

// MPI parallelize the second dimension -- # of mesh nodes 
// abs or rel
// argv[1]: error tolerance
// argv[2]: number of timesteps
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, np_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np_size);

	std::chrono::steady_clock clock;
	std::chrono::steady_clock::time_point start, stop;
	std::chrono::steady_clock::duration duration;

    std::vector<double> tol = {1e10, 5e10, 1e11, 5e11, 1e12, 5e12, 1e13, 5e13};
    const char *suffix[8] = {"1e10", "5e10", "1e11", "5e11", "1e12", "5e12", "1e13", "5e13"};
    size_t timeSteps;
    bool rel = (strcmp(argv[1], "rel") == 0);
    if (rel) {
        tol.at(rank) = log(1+tol.at(rank));
    }
    char datapath[2048], filename[2048], readin_f[2048], write_f[2048];
    strcpy(datapath, argv[2]);
    strcpy(filename, argv[3]);
    timeSteps = atoi(argv[4]);
    unsigned char *compressed_data = 0;
    std::vector<std::size_t> shape(4);
    if (rel) std::cout << "relative eb = " << tol.at(rank) << "\n";
    else std::cout << "absolute eb = " << tol.at(rank) << "\n";
    std::cout << "number of timeSteps: " << timeSteps << "\n"; 

    start = clock.now();
    adios2::ADIOS ad;
    adios2::IO reader_io = ad.DeclareIO("XGC");
    adios2::IO read_vol_io = ad.DeclareIO("xgc_vol");

    char vol_file[2048];
    sprintf(vol_file, "%sxgc.f0.mesh.bp", datapath);
    std::cout << vol_file << "\n";
    adios2::Engine reader_vol = read_vol_io.Open(vol_file, adios2::Mode::Read);
    adios2::Variable<double> var_i_f_in;
    var_i_f_in = read_vol_io.InquireVariable<double>("f0_grid_vol_vonly");
    size_t nnodes = var_i_f_in.Shape()[1];
    var_i_f_in.SetSelection(adios2::Box<adios2::Dims>({0, 0}, {1, nnodes}));
    std::vector<double> grid_vol;
    reader_vol.Get<double>(var_i_f_in, grid_vol);
    reader_vol.Close();
    sprintf(write_f, "%s%s%s", filename, ".mgard.4d.", suffix[rank]);

    double FSL = std::accumulate(grid_vol.begin(), grid_vol.end(), 0.0);
    double div = 1.0/FSL;
    std::vector<double>coords_z(nnodes, div); 
    coords_z.at(0) = grid_vol.at(0) * 0.5 * div;
    for (size_t i=1; i<nnodes; i++)
        coords_z.at(i) = coords_z.at(i-1) + (grid_vol.at(i) + grid_vol.at(i-1)) * 0.5 * div;
    size_t vx=45, vy=45;
    double base_x = (1.0-1.0/vx)/(vx-1), base_y=(1.0-1.0/vy)/(vy-1), base_t=1.0/timeSteps;
    std::vector<double>coords_x(vx, 0.0), coords_y(vy, 0.0), coords_t(timeSteps, 0.0);
    coords_x.at(0) = 0.5/vx;
    coords_y.at(0) = 0.5/vy;
    coords_t.at(0) = 0.5/timeSteps;
    for (size_t idx=1; idx<vx; idx++) coords_x.at(idx) = base_x * idx + coords_x.at(0);
    for (size_t idx=1; idx<vy; idx++) coords_y.at(idx) = base_y * idx + coords_y.at(0);
    for (size_t idx=1; idx<timeSteps; idx++) coords_t.at(idx) = base_t * idx + coords_t.at(0);
    std::cout << "max x: " << coords_x.at(vx-1) << ", max y: " << coords_y.at(vy-1) << ", max z: " << coords_z.at(nnodes-1) << "\n";
    size_t temp_sz = nnodes * vx * vy; 
    double *i_f_5d = new double[temp_sz * timeSteps];

    for (size_t ts = 0; ts < timeSteps; ts ++) {
        if (ts==0)
            sprintf(readin_f, "%s%s", datapath, filename);
        else {
            char ts_fn[2048];
            strncpy(ts_fn, filename, strlen(filename)-5); 
            sprintf(readin_f, "%s%s%d.bp", datapath, ts_fn, ts*10);
        }
        adios2::Engine reader = reader_io.Open(readin_f, adios2::Mode::Read);
        std::cout << "readin: " << readin_f << "\n";
        // Inquire variable
        var_i_f_in = reader_io.InquireVariable<double>("i_f");

        if (ts==0) {
            shape = var_i_f_in.Shape();
            std::cout << "processor " << rank << " read " << readin_f << " frome: {0, 0, 0, 0} for {";
            std::cout << 1 << ", " << shape[1] << ", " << shape[2] << ", " << shape[3] << "}.\n";
        }
        var_i_f_in.SetSelection(adios2::Box<adios2::Dims>({0, 0, 0, 0}, {1, shape[1], shape[2], shape[3]}));
        std::vector<double> i_f;
        reader.Get<double>(var_i_f_in, i_f);
        reader.Close();

        memcpy(&i_f_5d[ts*temp_sz], i_f.data(), i_f.size()*sizeof(double));
        std::cout << "rank " << rank << " readin size: " << i_f.size() << "\n";
    }
    if (rel) {
        for (size_t it=0; it < temp_sz ; it++) 
            i_f_5d[it] = log(i_f_5d[it]);
    }
    std::cout << "begin compression...\n";
    if (timeSteps==1) {
        const std::array<std::vector<double>, 3> coords = {coords_y, coords_z, coords_x};
        const std::array<std::size_t, 3> dims = {shape[1], shape[2], shape[3]};
        const mgard::TensorMeshHierarchy<3, double> hierarchy(dims, coords);
        const size_t ndof = hierarchy.ndof();
        const mgard::CompressedDataset<3, double> compressed = mgard::compress(hierarchy, i_f_5d, 0.0, tol.at(rank));
        std::cout << "processor " << rank << ", after compression: " << compressed.size() << "\n";
        const mgard::DecompressedDataset<3, double> decompressed = mgard::decompress(compressed);
        FileWriter_ad(write_f, ((double *)decompressed.data()), {timeSteps, shape[1], shape[2], shape[3]}, rel);
    } else {
        const std::array<std::vector<double>, 4> coords = {coords_t, coords_y, coords_z, coords_x};
        const std::array<std::size_t, 4> dims = {timeSteps, shape[1], shape[2], shape[3]};
        const mgard::TensorMeshHierarchy<4, double> hierarchy(dims, coords);
        const size_t ndof = hierarchy.ndof();
        const mgard::CompressedDataset<4, double> compressed = mgard::compress(hierarchy, i_f_5d, 0.0, tol.at(rank));
        std::cout << "Processor " << rank << ", after compression: " << compressed.size() << "\n"; 
        const mgard::DecompressedDataset<4, double> decompressed = mgard::decompress(compressed); 
        FileWriter_ad(write_f, ((double *)decompressed.data()), {timeSteps, shape[1], shape[2], shape[3]}, rel);
    }
    stop = clock.now();
    duration = stop - start;
    std::cout << "Processor " << rank << ": compression: " << std::floor(SECONDS(duration)/60.0) << "min and " << SECONDS(duration)%60 << "sec" << std::endl;

    delete i_f_5d;
    MPI_Finalize();
}
