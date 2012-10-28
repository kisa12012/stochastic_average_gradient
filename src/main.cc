#include <fstream>
#include <sstream>
#include <iostream>

#include "lcsgd.h"
#include "sgd.h"
#include "cmdline.h"

using namespace std;
using namespace lcsgd;

namespace {
data_t ParseFile(const std::string& file_path) {
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
  cmdline::parser cmdl;

  cmdl.add<std::string>("train_file", 't', "train file path (train)", true, "");
  // cmdl.add<std::string>("algorithm", 'a', "algorithm (SAG/SGD)", false, "SAG");
  cmdl.add<double>("alpha", 'A', "alpha", false, 1.0);
  cmdl.add<double>("lambda", 'L', "lambda", false, 0.01);
  cmdl.parse_check(argc, argv);

  data_t data = ParseFile(cmdl.get<std::string>("train_file"));

  SGD sgd;
  sgd.LoadData(data);
  sgd.SetAlpha(cmdl.get<double>("alpha"));
  sgd.SetLambda(cmdl.get<double>("lambda"));
  LCSGD lcsgd;
  lcsgd.LoadData(data);
  lcsgd.SetAlpha(cmdl.get<double>("alpha"));
  lcsgd.SetLambda(cmdl.get<double>("lambda"));

  std::cout << "*** SGD *** \t *** SAG ***" << std::endl;
  for (int i = 0; i < 100000; ++i) {
    sgd.Update(100);
    lcsgd.Update(100);
    std::cout << sgd.Evaluation() << "\t\t" << lcsgd.Evaluation() << std::endl;
  }
  return 0;
}
