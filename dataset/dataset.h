#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

namespace {
struct split {
    enum empties_t { empties_ok, no_empties };
};
}

std::vector<std::string>& split_string(std::vector<std::string>&             result,
                                       const std::string&                         s,
                                       const std::string&                     delim,
                                       split::empties_t empties = split::empties_ok) {
    result.clear();
    size_t current;
    size_t next = -1;
    do {
        if (empties == split::no_empties) {
            next = s.find_first_not_of( delim, next + 1 );
            if (next == std::string::npos) break;
            next -= 1;
        }
        current = next + 1;
        next = s.find_first_of( delim, current );
        result.push_back( s.substr( current, next - current ) );
    }
    while (next != std::string::npos);
    return result;
}

class dataset {
	public:
		dataset() {}

		dataset(const char* filename) {
			std::ifstream ifstr(filename);
			if (!ifstr.is_open()) { throw "Could not open file."; }
			rows.clear();
			std::string line;
			while (std::getline(ifstr, line)) {
				std::vector<std::string> fields;
				split_string(fields, line, " ");
				std::vector<double> row;
				for (auto f : fields) {
					double v = atof(f.c_str());
					row.push_back(v);
				}
				rows.push_back(row);
			}
		}

		// scale values to the interval [-1,1]
		// WARNING: original values will be lost
		void normalize() { 
			double min = 0, max = 0;
			// first pass to determine min and max
			for (auto & row : rows) {
				for (auto v : row) {
					if (min > v) min = v;
					if (max < v) max = v;
				}
			}
			// second pass to normalize
			for (auto & row : rows) {
				for (auto & v : row) {
					v = 2 * (v - min) / max - min - 1;
				}
			}
		}

		double print() {
			for (auto & row : rows) {
				for (auto v : row) {
					std::cout << v << " ";
				}
				std::cout << std::endl;
			}
		}

    dataset(const std::vector<std::vector<double>>& rows_) : rows(rows_) {}
    std::vector<std::vector<double>> rows;
};

#endif // DATASET_HPP
