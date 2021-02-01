#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>

#include "adios2.h"

template <typename Type>
void FileWriter_bin(const char *filename, Type *data, size_t size)
{
  std::ofstream fout(filename, std::ios::binary);
  fout.write((const char*)(data), size*sizeof(Type));
  fout.close();
}

int main(int argc, char **argv) {
  adios2::ADIOS ad;
  adios2::IO reader_io = ad.DeclareIO("XGC");

  adios2::Engine reader = reader_io.Open("decompressed_5d_2ts.bp", adios2::Mode::Read);

  // Inquire variable
  adios2::Variable<double> var_i_f_in;

  var_i_f_in = reader_io.InquireVariable<double>("i_f_5d");

  std::vector<std::size_t> shape = var_i_f_in.Shape();;

  var_i_f_in.SetSelection(adios2::Box<adios2::Dims>(
                        {0, 0, 0, 0, 0}, {shape[0],  shape[1], shape[2], shape[3], shape[4]}));
  std::vector<double> i_f;
  reader.Get<double>(var_i_f_in, i_f);

  reader.Close();

  FileWriter_bin("decompressed_5d_2ts.bin", (double *)i_f.data(), shape[0]*shape[1]*shape[2]*shape[3]*shape[4]);

}
