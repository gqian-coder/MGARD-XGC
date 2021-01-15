#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>

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

int main(int argc, char **argv) {
	std::chrono::steady_clock clock;
	std::chrono::steady_clock::time_point start, stop;
	std::chrono::steady_clock::duration duration;

  double tol, result;
  int out_size;
  unsigned char *compressed_data = 0;
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
  adios2::IO reader_io = ad.DeclareIO("XGC");

  start = clock.now();
  adios2::Engine reader = reader_io.Open("/gpfs/alpine/proj-shared/csc143/gongq/MReduction/MGARDx/build/test/data/xgc.f0.00420.bp", adios2::Mode::Read);
  stop = clock.now();
  duration = stop - start;
  std::cout << "constructed reader (" << SECONDS(duration) << " s)" << std::endl;;

  // Inquire variable
  adios2::Variable<double> var_i_f_in;

  start = clock.now();
  var_i_f_in = reader_io.InquireVariable<double>("i_f");
  stop = clock.now();
  duration = stop - start;
  std::cout << "inquired about i_f (" << SECONDS(duration) << " s)" << std::endl;

  std::vector<std::size_t> shape = var_i_f_in.Shape();;

  std::cout << shape[0] << " " << shape[1] << " "
            << shape[2] << " " << shape[3] << "\n";

  start = clock.now();
  var_i_f_in.SetSelection(adios2::Box<adios2::Dims>(
                        {0, 0, 0, 0}, {shape[0],  shape[1], shape[2], shape[3]}));
  stop = clock.now();
  duration = stop - start;
  std::cout << "set selection (" << SECONDS(duration) << " s)" << std::endl;

  std::vector<double> i_f;
  start = clock.now();
  reader.Get<double>(var_i_f_in, i_f);
  stop = clock.now();
  duration = stop - start;
  std::cout << "got variable (" << SECONDS(duration) << " s)" << std::endl;

  start = clock.now();
  reader.Close();
  stop = clock.now();
  duration = stop - start;
  std::cout << "closed reader (" << SECONDS(duration) << " s)" << std::endl;

  size_t ns = shape[2];
//  const std::array<std::size_t, 1> dims = {u_global_size};
  std::cout << "readin: {" << shape[0] << ", " << shape[1] << ", " << ns << ", " << shape[3] << "}\n";
  const std::array<std::size_t, 4> dims = {shape[0], shape[1], ns, shape[3]};
  std::cout << "begin compression...\n";

  const mgard::TensorMeshHierarchy<4, double> hierarchy(dims);
  const size_t ndof = hierarchy.ndof();
  tol = 1e13;
  std::cout << "err tolerance: " << tol << "\n";
  start = clock.now();
  const mgard::CompressedDataset<4, double> compressed =
	  mgard::compress(hierarchy, i_f.data(), 0.0, tol);
  stop = clock.now();
  duration = stop - start;
  const double throughput = static_cast<double>(sizeof(double) * ndof) / (1 << 20) / SECONDS(duration);
  std::cout << "Compression: " << std::floor(SECONDS(duration)/60.0) << "min and " << SECONDS(duration)%60 << "sec, and throughput = "; 
  std::cout << throughput << " MiB / s" << std::endl;
  std::cout << "after compression: " << compressed.size() << "\n"; 

  FileWriter_bin("compressed.bin", (unsigned char *)compressed.data(), compressed.size());

  stop = clock.now();
  duration = stop - start;
  const mgard::DecompressedDataset<4, double> decompressed = mgard::decompress(compressed); 
  const double throughput_d = static_cast<double>(sizeof(double) * ndof) / (1 << 20) / SECONDS(duration);
  std::cout << "Decompression: " << std::floor(SECONDS(duration)/60.0) << "min and " << SECONDS(duration)%60 << "sec, and throughput = ";
  std::cout  << throughput_d << " MiB / s" << std::endl;
  FileWriter_bin("decompressed.bin", (double *)decompressed.data(), shape[0]*shape[1]*ns*shape[3]);

  FileWriter_ad("decompressed.bp", (double *)decompressed.data(), {shape[0], shape[1], ns, shape[3]});

}
