#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>

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
void FileWriter_ad(const char *filename, Type *data, std::vector<size_t> size)
{
    adios2::ADIOS ad;
    adios2::IO bpIO = ad.DeclareIO("WriteBP_File");
    adios2::Variable<Type> bp_fdata = bpIO.DefineVariable<Type>(
        "i_f", size, {0,0,0,0}, size,  adios2::ConstantDims);
    // Engine derived class, spawned to start IO operations //
    adios2::Engine bpFileWriter = bpIO.Open(filename, adios2::Mode::Write);
    bpFileWriter.Put<Type>(bp_fdata, data);
    bpFileWriter.Close();
}

template<typename Type>
const Type* mgard_reconstruct(Type *data, std::vector<size_t> shape)
{
    const std::array<std::size_t, 4> dims = {shape[0], shape[1], shape[2], shape[3]};
    const mgard::TensorMeshHierarchy<4, double> hierarchy(dims);
    const size_t ndof = hierarchy.ndof();
    double tol = 1e13;
    const mgard::CompressedDataset<4, double> compressed =
        mgard::compress(hierarchy, data, 0.0, tol);
    size_t num_elem = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<double>());
    const mgard::DecompressedDataset<4, Type> decompressed = mgard::decompress(compressed);
    return decompressed.data();
}

int main(int argc, char **argv) {
	std::chrono::steady_clock clock;
	std::chrono::steady_clock::time_point start, stop;
	std::chrono::steady_clock::duration duration;

    double tol, result;
#if 0
    MPI_Init(&argc, &argv);
    int rank, comm_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm comm = MPI_COMM_SELF;

    if (argc != 3) {
        cerr << "Wrong arugments!\n";
        return 1;
    }
#endif
    adios2::ADIOS ad;
    // Reader I/0
    adios2::IO reader_io = ad.DeclareIO("XGC");

    adios2::Engine reader = reader_io.Open("../untwisted_xgc.f0.00400.bp", adios2::Mode::Read);
    adios2::Variable<double> var_i_f_in;
    // Writer i/O
    adios2::IO bpIO = ad.DeclareIO("WriteBP_File");
    adios2::Variable<double> var_i_f_out = bpIO.DefineVariable<double>("rct_i_f", {}, {}, {8},  adios2::ConstantDims); 
    adios2::Engine writer = bpIO.Open("../xgc.f0.00400.mgard.bp", adios2::Mode::Write); 
    // read data
    size_t istep=0;
    size_t rct_sec = 0;
    size_t tot_elem = 0;
    while (true) {
        auto status = reader.BeginStep();
        if (status == adios2::StepStatus::OK) {
            var_i_f_in = reader_io.InquireVariable<double>("i_f");
            auto shape = var_i_f_in.Shape();
            std::cout << "step " << istep << " read: {" << shape[0] << ",";
            std::cout << shape[1] << "," << shape[2] << "," << shape[3] << "}\n";
            istep ++;
            if (shape[1] < 2)
                continue;
            var_i_f_in.SetSelection(adios2::Box<adios2::Dims>(
                {0, 0, 0, 0}, {shape[0],  shape[1], shape[2], shape[3]}));
            std::vector<double> i_f;
            reader.Get<double>(var_i_f_in, i_f);
            reader.PerformGets();
        
            tot_elem += std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<double>());
            start = clock.now();
            // MGARD Compression
            auto reconstructed = mgard_reconstruct(i_f.data(), shape);
            stop = clock.now();
            rct_sec += SECONDS(stop - start); 
            writer.BeginStep();
            writer.Put(var_i_f_out, reconstructed); 
            writer.EndStep();
        } else if (status == adios2::StepStatus::EndOfStream) {
            std::cout << "End of stream" << std::endl;
            break;
        }
        reader.EndStep();
    }
    reader.Close();
    writer.Close();
    const double throughput_d = static_cast<double>(sizeof(double) * tot_elem) / (1 << 20) / rct_sec; 
    std::cout << "Total reconstruction cost: " << std::floor(SECONDS(duration)/60.0) << "min and " << SECONDS(duration)%60 << "sec, and throughput = ";
    std::cout  << throughput_d << " MiB / s" << std::endl;

}
