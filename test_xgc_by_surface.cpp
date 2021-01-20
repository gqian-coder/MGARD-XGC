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
const double* mgard_reconstruct_3D(Type *data, std::vector<size_t> shape, size_t timeStep)
{
    const std::array<size_t, 3> dims = {shape[0], shape[1], shape[2]};
    const mgard::TensorMeshHierarchy<3, double> hierarchy(dims);
    const size_t ndof = hierarchy.ndof();
    double tol = 1e13;
    const mgard::CompressedDataset<3, double> compressed =
        mgard::compress(hierarchy, data, 0.0, tol);
    std::cout << "after compression: " << compressed.size() << "\n";
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
    std::cout << "after compression: " << compressed.size() << "\n";
    const mgard::DecompressedDataset<4, double> decompressed = mgard::decompress(compressed);
//    FileWriter_ad(("i_f.mgard." + std::to_string(timeStep) + ".bp").c_str(), (double *)decompressed.data(), shape);
    return decompressed.data();
}


int main(int argc, char **argv) {
	std::chrono::steady_clock clock;
	std::chrono::steady_clock::time_point start, stop;
	std::chrono::steady_clock::duration duration;

    double tol, result;
    adios2::ADIOS ad;
//    adios2::ADIOS ad_w;
    // Reader I/0
    adios2::IO reader_io = ad.DeclareIO("XGC");
    adios2::Engine reader = reader_io.Open("/gpfs/alpine/proj-shared/csc143/gongq/andes/MReduction/MGARD-XGC/untwisted_xgc.f0.00400.bp", adios2::Mode::Read); 
    adios2::Variable<double> var_i_f_in;
    // Writer i/O
    adios2::IO writer_io = ad.DeclareIO("Output"); 
    adios2::Engine writer = writer_io.Open("i_f.mgard.bp", adios2::Mode::Write); 
    adios2::Variable<double> var_i_f_out = writer_io.DefineVariable<double>("rct_i_f", {}, {}, {adios2::UnknownDim}); 

    // read data
    size_t rct_sec = 0;
    size_t tot_elem = 0;
    adios2::fstep iStep;
    tol = 1e13;
    for (unsigned int timeStep = 0; timeStep < 497; ++timeStep) { 
        reader.BeginStep();
        var_i_f_in = reader_io.InquireVariable<double>("i_f");
        std::vector<std::size_t> shape = var_i_f_in.Shape();
        var_i_f_in.SetSelection(adios2::Box<adios2::Dims>(
                    {0, 0, 0, 0}, shape));//{shape[0],  shape[1], shape[2], shape[3]}));
        std::vector<double>i_f_in;
        reader.Get<double>(var_i_f_in, i_f_in); 
        reader.EndStep();
        size_t num_nz = 0;
        std::cout << "step " << timeStep << " readin: " << i_f_in.size()/39/39/8 << " nodes \n";
        for (unsigned int it=0; it<i_f_in.size(); it++) {
            num_nz += (i_f_in[it]==0); 
        }
        std::cout << "step " << timeStep << ": number of zeros = " << num_nz << "\n";
        tot_elem += std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<double>());
        start = clock.now();
        std::cout << "step " << timeStep << " read: {" << shape[0] << ",";
        std::cout << shape[1] << "," << shape[2] << "," << shape[3] << "}\n";
        if (shape[1] < 2) {
            for (unsigned int k = 0; k < shape[1]; k ++) {
                const std::array<size_t, 3> dims = {shape[0], shape[2], shape[3]};
                const mgard::TensorMeshHierarchy<3, double> hierarchy(dims);
                const mgard::CompressedDataset<3, double> compressed =
                mgard::compress(hierarchy, i_f_in.data(), 0.0, tol);
                std::cout << "after compression: " << compressed.size() << "\n";
                const mgard::DecompressedDataset<3, double> decompressed = mgard::decompress(compressed);
                writer.BeginStep();
                var_i_f_out.SetSelection(adios2::Box<adios2::Dims>({}, {shape[0]*shape[1]*shape[2]*shape[3]}));
                writer.Put<double>(var_i_f_out, (double *)decompressed.data());
                writer.EndStep();
            }
        } else {
//            const double *reconstructed = mgard_reconstruct_4D(i_f_in.data(), shape, timeStep);
            const std::array<size_t, 4> dims = {shape[0], shape[1], shape[2], shape[3]};
            const mgard::TensorMeshHierarchy<4, double> hierarchy(dims);
            const mgard::CompressedDataset<4, double> compressed =
            mgard::compress(hierarchy, i_f_in.data(), 0.0, tol);
            std::cout << "after compression: " << compressed.size() << "\n";
            const mgard::DecompressedDataset<4, double> decompressed = mgard::decompress(compressed);
            writer.BeginStep();
            var_i_f_out.SetSelection(adios2::Box<adios2::Dims>({}, {shape[0]*shape[1]*shape[2]*shape[3]}));
            writer.Put<double>(var_i_f_out, (double *)decompressed.data());
            writer.EndStep();
        }
        // MGARD Compression
        stop = clock.now();
        rct_sec += SECONDS(stop - start); 
    } 
    reader.Close();
//    writer.Close();
    const double throughput_d = static_cast<double>(sizeof(double) * tot_elem) / (1 << 20) / rct_sec; 
    std::cout << "Total reconstruction cost: " << std::floor(rct_sec/60.0) << " min and " << rct_sec%60 << " sec, and throughput = ";
    std::cout  << throughput_d << " MiB / s" << std::endl;

}
