#include "lcsgd.h"

#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;
using namespace lcsgd;

namespace {
data_t ParseFile(const char* file_path) {
  data_t data;
  
  ifstream ifs(file_path);
  if (!ifs) {
    cerr << "cannot open " << file_path << endl;
    return data;
  }

  size_t lineN = 0;
  for (string line; getline(ifs, line); ++lineN) {
    datum_t datum;
    std::istringstream iss(line);

    if (!(iss >> datum.label)) {
      std::cerr << "parse error: you must set category in line " << lineN << std::endl;
      continue;
    }

    int id = 0;
    char comma = 0;
    double value = 0.0;
    while (iss >> id >> comma >> value) {
      if (id > data.max_feature_id)
        data.max_feature_id = id;
      datum.features.emplace_back(id, value);
    }

    data.examples.emplace_back(datum);
    ++data.data_size;
  }

  return data;
}
} //namespace {anonymous}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " [training file]" << std::endl;
    return -1;
  }

  LCSGD lcsgd;
  data_t data = ParseFile(argv[1]);
  std::cout << "segf" << std::endl;
  lcsgd.LoadData(data);

  lcsgd.Update(1000);
  return 0;
}
