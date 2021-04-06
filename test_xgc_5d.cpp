#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

#include "adios2.h"
#include "mgard/mgard_api.h"

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
    std::vector<size_t>local_dim, size_t para_dim, bool rel)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rel) {
        size_t tot_sz = local_dim[0] * local_dim[1] * local_dim[2] * local_dim[3] * local_dim[4];
        for (size_t it=0; it<tot_sz; it++)
            data[it] = exp(data[it]);
    }
    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO bpIO = ad.DeclareIO("WriteBP_File");
    std::cout << "processor " << rank << ": {" << global_dim[0] << ", " << global_dim[1] << ", " << global_dim[2] << ", " << global_dim[3] << ", " << global_dim[4] << "}, {";
    std::cout << local_dim[0] << ", " <<local_dim[1] << ", " << local_dim[2] << ", " << local_dim[3] << ", " << local_dim[4] << "}, " << para_dim*rank << "\n";
    adios2::Variable<Type> bp_fdata = bpIO.DefineVariable<Type>(
          "i_f_5d", global_dim, {0, 0, 0, para_dim*rank, 0}, local_dim,  adios2::ConstantDims);
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

    double tol;
    size_t timeSteps, temp_dim, temp_sz, local_dim, local_sz;
    bool rel = (strcmp(argv[1], "rel") == 0);
    tol = atof(argv[2]);
    if (rel) tol = log(1+tol);
    char datapath[2048], filename[2048], readin_f[2048], write_f[2048];
    strcpy(datapath, argv[3]);
    strcpy(filename, argv[4]);
    timeSteps = atoi(argv[5]);
    unsigned char *compressed_data = 0;
    double *i_f_5d;
    std::vector<std::size_t> shape(4);
    if (rank==0) {
        if (rel) std::cout << "relative eb = " << tol << "\n";
        else std::cout << "absolute eb = " << tol << "\n";
        std::cout << "number of timeSteps: " << timeSteps << "\n"; 
    }

    start = clock.now();
    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO reader_io = ad.DeclareIO("XGC");

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
        adios2::Variable<double> var_i_f_in;
        var_i_f_in = reader_io.InquireVariable<double>("i_f");

        if (ts==0) {
            shape = var_i_f_in.Shape();
            temp_dim  = (size_t)ceil((float)shape[2]/np_size);
            local_dim = ((rank==np_size-1) ? (shape[2]-temp_dim*rank) : temp_dim);
            temp_sz   = temp_dim  * shape[0]*shape[1]*shape[3];
            local_sz  = local_dim * shape[0]*shape[1]*shape[3]; 
            i_f_5d = new double[temp_sz * timeSteps];
            if (rank==0) 
                std::cout << "global shape: {" << shape[0] << ", " << shape[1] << ", " << shape[2] << ", " << shape[3] << "}, ";
        }

        size_t start_pos = (rank==0) ? 0 : temp_dim*rank;
        size_t read_node = local_dim;
        if (ts==0) {
            std::cout << "processor " << rank << " read " << readin_f << " frome: {0, 0, " << temp_dim*rank << ", 0} for {";
            std::cout << shape[0] << ", " << shape[1] << ", " << local_dim << ", " << shape[3] << "}.\n";
        }
        var_i_f_in.SetSelection(adios2::Box<adios2::Dims>({0, 0, start_pos, 0}, {shape[0], shape[1], read_node, shape[3]}));
        std::vector<double> i_f;
        reader.Get<double>(var_i_f_in, i_f);
        reader.Close();

        memcpy(&i_f_5d[ts*temp_sz], i_f.data(), i_f.size()*sizeof(double));
        std::cout << "rank " << rank << " readin size: " << i_f.size() << "\n";
    }
    if (rel) {
        for (size_t it=0; it < temp_sz * timeSteps; it++) 
            i_f_5d[it] = log(i_f_5d[it]);
    }
    std::cout << "begin compression...\n";
    sprintf(write_f, "%s%s", filename, ".mgard.1e12"); 
    if (timeSteps==1) {
        const std::array<std::size_t, 4> dims = {shape[0], shape[1], local_dim, shape[3]};
        const mgard::TensorMeshHierarchy<4, double> hierarchy(dims);
        const size_t ndof = hierarchy.ndof();
        const mgard::CompressedDataset<4, double> compressed = mgard::compress(hierarchy, i_f_5d, 0.0, tol);
        std::cout << "processor " << rank << ", after compression: " << compressed.size() << "\n";
        const mgard::DecompressedDataset<4, double> decompressed = mgard::decompress(compressed);
        FileWriter_ad(write_f, ((double *)decompressed.data()), {timeSteps, shape[0], shape[1], shape[2], shape[3]}, {timeSteps, shape[0], shape[1], local_dim, shape[3]}, temp_dim, rel);
    } else {
        const std::array<std::size_t, 5> dims = {timeSteps, shape[0], shape[1], local_dim, shape[3]};
        const mgard::TensorMeshHierarchy<5, double> hierarchy(dims);
        const size_t ndof = hierarchy.ndof();
        const mgard::CompressedDataset<5, double> compressed = mgard::compress(hierarchy, i_f_5d, 0.0, tol);
        std::cout << "Processor " << rank << ", after compression: " << compressed.size() << "\n"; 
        const mgard::DecompressedDataset<5, double> decompressed = mgard::decompress(compressed); 
        FileWriter_ad(write_f, ((double *)decompressed.data()), {timeSteps, shape[0], shape[1], shape[2], shape[3]}, {timeSteps, shape[0], shape[1], local_dim, shape[3]}, temp_dim, rel);
    }
    stop = clock.now();
    duration = stop - start;
    std::cout << "Processor " << rank << ": compression: " << std::floor(SECONDS(duration)/60.0) << "min and " << SECONDS(duration)%60 << "sec" << std::endl;

    delete i_f_5d;
    MPI_Finalize();
}
