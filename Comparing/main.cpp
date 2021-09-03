#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
//#include <opencv2/core/utility.hpp>


using namespace std;
//using namespace cv;

using TYPE = double;

//------------------------------------------------
template< typename T > vector< vector<T> > loadtxt( const string &filename )
{
    vector< vector<T> > data;
    ifstream in( filename );
    for ( string line; getline( in, line ); )
    {
        stringstream ss( line );
        vector<T> row;
        for ( T d; ss >> d; ) row.push_back( trunc(d*1)/1.0 );
        data.push_back( row );
    }
    return data;
}

//------------------------------------------------
template< typename T > void print( const vector< vector<T> > &data, ostream &out = cout)
{
    for ( auto &row : data )
    {
        for ( auto &item : row ) out << item << "\t";
        out << '\n';
    }
}

//------------------------------------------------
template< typename T > void concat( vector< vector<T> > &vector1, const vector< vector<T> > &vector2 )
{
    vector1.insert( vector1.end(), vector2.begin(), vector2.end() );
}

template< typename T > void comparing( vector< vector<T> > &data, vector< size_t > &range )
{
	// Para cada metodo i 
	for (size_t method_i = 0; method_i < range.size() - 2; method_i++) {
		// Comparamos aos metodos j subsequentes
		for (size_t method_j = method_i + 1; method_j < range.size() - 1; method_j++) {
			// Todos os pontos do método i
			for (size_t i_idx = range[method_i]; i_idx < range[method_i + 1]; i_idx++) {
				auto i_x = data[i_idx][0], i_y = data[i_idx][1];
				// Com todos os pontos do método j
				for (size_t j_idx = range[method_j]; j_idx < range[method_j + 1]; j_idx++) {
					auto j_x = data[j_idx][0], j_y = data[j_idx][1];
					// Se os métodos i e j detectam um mesmo ponto
					if (i_x == j_x && i_y == j_y) {
						// Registramos a detecção conjunta para o ponto no método i
						if (data[i_idx].size() == 2) {
							data[i_idx].push_back(2);
							for (size_t k = 0; k < range.size() - 1; k++)
								data[i_idx].push_back(-1);
							data[i_idx][method_i + 3] = i_idx - range[method_i];
							data[i_idx][method_j + 3] = j_idx - range[method_j];
						} // Ou atualizamos um registro anterior
						else { 
							data[i_idx][2] += 1;
							data[i_idx][method_j + 3] = j_idx - range[method_j];
						}
						// O mesmo vale para o ponto no método j
						if (data[j_idx].size() == 2) {
							data[j_idx].push_back(2);
							for (size_t k = 0; k < range.size() - 1; k++)
								data[j_idx].push_back(-1);
							data[j_idx][method_i + 3] = i_idx - range[method_i];
							data[j_idx][method_j + 3] = j_idx - range[method_j];
						} else {
							data[j_idx][2] += 1;
							data[j_idx][method_i + 3] = i_idx - range[method_i];
						}
						// E passamos ao proximo ponto do método i
						break;
					}
				}
			}
		}
	}
	// Identificando pontos chave são exclusivos de um método
	for (size_t method_i = 0; method_i < range.size() - 1; method_i++) {
		for (size_t i_idx = range[method_i]; i_idx < range[method_i + 1]; i_idx++) {
			if (data[i_idx].size() == 2) {
				data[i_idx].push_back(1);
				for (size_t k = 0; k < range.size() - 1; k++)
					data[i_idx].push_back(-1);
				data[i_idx][method_i + 3] = i_idx - range[method_i];
			}
		}
	}
}

//======================================================================
int main(int argc, char* argv[])
{
    
    //CommandLineParser parser(argc, argv,
    //                         "{@AKAZE | kpts1_Akaze.txt | Input Akaze Matches}"
    //                         "{@ORB | kpts1_ORB.txt | Input ORB Matches}"
    //                         "{@SIFT | kpts1_SIFT.txt | Input SIFT Matches}"
    //                         "{@SURF | kpts1_SURF.txt | Input SURF Matches}");
    
    //auto data = loadtxt<TYPE>( parser.get<String>("@AKAZE").c_str() );
    auto data = loadtxt<TYPE>( "kpts1_Akaze.txt" );
    auto akaze_end = data.size();
    //auto temp = loadtxt<TYPE>( parser.get<String>("@ORB").c_str() );
    auto temp = loadtxt<TYPE>( "kpts1_ORB.txt" );
    concat(data,temp);
    auto orb_end = data.size();
    //temp = loadtxt<TYPE>( parser.get<String>("@SIFT").c_str() );
    temp = loadtxt<TYPE>( "kpts1_SIFT.txt" );
    concat(data,temp);
    auto sift_end = data.size();
    //temp = loadtxt<TYPE>( parser.get<String>("@SURF").c_str() );
    temp = loadtxt<TYPE>( "kpts1_SURF.txt" );
    concat(data,temp);
    auto surf_end = data.size();
    
    vector<size_t> range = {0, akaze_end, orb_end, sift_end, surf_end};
    
    comparing(data, range);
    
    ofstream out( "saida.txt" );
    print( data , out );
    out.close();
}

