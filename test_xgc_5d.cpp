#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

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
    std::vector<size_t>local_dim, size_t para_dim)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO bpIO = ad.DeclareIO("WriteBP_File");
    std::cout << "processor " << rank << ": {" << global_dim[0] << ", " << global_dim[1] << ", " << global_dim[2] << ", " << global_dim[3] << ", " << global_dim[4] << "}, {";
    std::cout << local_dim[0] << ", " <<local_dim[1] << ", " << local_dim[2] << ", " << local_dim[3] << ", " << local_dim[4] << "}, " << para_dim*rank << "\n";
    adios2::Variable<Type> bp_fdata = bpIO.DefineVariable<Type>(
          "i_f_5d", global_dim, {0, para_dim*rank ,0, 0, 0}, local_dim,  adios2::ConstantDims);
    // Engine derived class, spawned to start IO operations //
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
// argv[1]: error tolerance
// argv[2]: number of timesteps
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	std::chrono::steady_clock clock;
	std::chrono::steady_clock::time_point start, stop;
	std::chrono::steady_clock::duration duration;

    double tol;
    size_t timeSteps, temp_dim, temp_sz, local_dim, local_sz;
    tol = atof(argv[1]);
    timeSteps = atoi(argv[2]);
    unsigned char *compressed_data = 0;
    double *i_f_5d;
    char filename[2048];
    std::vector<std::size_t> shape(4);
    if (rank==0)
        std::cout << "err tolerance: " << tol << "\n";

    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO reader_io = ad.DeclareIO("XGC");

    for (size_t ts = 0; ts < timeSteps; ts ++) {
        if (ts==0)
            sprintf(filename, "/gpfs/alpine/proj-shared/csc143/gongq/XGC/d3d_coarse_v2/untwisted_4D_600.bp"); 
        else
            sprintf(filename, "/gpfs/alpine/proj-shared/csc143/gongq/XGC/d3d_coarse_v2/untwisted_4D_6%d.bp", ts*10);
        adios2::Engine reader = reader_io.Open(filename, adios2::Mode::Read);

        // Inquire variable
        adios2::Variable<double> var_i_f_in;
        var_i_f_in = reader_io.InquireVariable<double>("i_f");

        if (ts==0) {
            shape = var_i_f_in.Shape();
            temp_dim = (size_t)ceil((float)shape[1]/size);
            temp_sz  = temp_dim*shape[0]*shape[2]*shape[3];
            local_dim = ((rank==size-1) ? (shape[1]-temp_dim*rank) : temp_dim); 
            local_sz  = local_dim*shape[0]*shape[2]*shape[3];
            i_f_5d = new double[temp_sz * timeSteps];
            if (rank==0) {   
                std::cout << "global shape: {" << shape[0] << ", " << shape[1] << ", " << shape[2] << ", " << shape[3] << "}, ";
            }
        }
        std::cout << "processor " << rank << " read " << filename << " frome: {0, " << temp_dim*rank << ", 0, 0} for {";
        std::cout << shape[0] << ", " << local_dim << ", " << shape[2] << ", " << shape[3] << "}.\n";
        var_i_f_in.SetSelection(adios2::Box<adios2::Dims>({0, temp_dim*rank, 0, 0}, {shape[0], local_dim, shape[2], shape[3]}));

        std::vector<double> i_f;
        reader.Get<double>(var_i_f_in, i_f);
        reader.Close();
        std::cout << "processor " << rank << ": offset = " << temp_sz*ts << "\n";
        memcpy(&i_f_5d[ts*temp_sz], i_f.data(), local_sz*sizeof(double));
    }
    std::cout << "begin compression...\n";
    if (timeSteps==1) {
        const std::array<std::size_t, 4> dims = {shape[0], local_dim, shape[2], shape[3]};
        const mgard::TensorMeshHierarchy<4, double> hierarchy(dims);
        const size_t ndof = hierarchy.ndof();
        const mgard::CompressedDataset<4, double> compressed = mgard::compress(hierarchy, i_f_5d, 0.0, tol);
        std::cout << "processor " << rank << ", after compression: " << compressed.size() << "\n";
        const mgard::DecompressedDataset<4, double> decompressed = mgard::decompress(compressed);
        FileWriter_ad("decompressed_5d.bp", (double *)decompressed.data(), {shape[0], shape[1], shape[2], shape[3], timeSteps}, {shape[0], local_dim, shape[2], shape[3], timeSteps}, temp_dim);
    } else {
        const std::array<std::size_t, 5> dims = {shape[0], local_dim, shape[2], shape[3], timeSteps}; 
        const mgard::TensorMeshHierarchy<5, double> hierarchy(dims);
        const size_t ndof = hierarchy.ndof();
//        start = clock.now();
        const mgard::CompressedDataset<5, double> compressed = mgard::compress(hierarchy, i_f_5d, 0.0, tol);
//    stop = clock.now();
//    duration = stop - start;
//    const double throughput = static_cast<double>(sizeof(double) * ndof) / (1 << 20) / SECONDS(duration);
//    std::cout << "Compression: " << std::floor(SECONDS(duration)/60.0) << "min and " << SECONDS(duration)%60 << "sec, and throughput = "; 
//    std::cout << throughput << " MiB / s" << std::endl;
        std::cout << "processor " << rank << ", after compression: " << compressed.size() << "\n"; 
//    FileWriter_bin("compressed.bin", (unsigned char *)compressed.data(), compressed.size());

        const mgard::DecompressedDataset<5, double> decompressed = mgard::decompress(compressed); 
        FileWriter_ad("decompressed_5d.bp", (double *)decompressed.data(), {shape[0], shape[1], shape[2], shape[3], timeSteps}, {shape[0], local_dim, shape[2], shape[3], timeSteps}, temp_dim);
    }
    delete i_f_5d;
    MPI_Finalize();
}
