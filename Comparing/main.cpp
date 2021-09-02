#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <opencv2/core/utility.hpp>


using namespace std;
using namespace cv;

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
        for ( T d; ss >> d; ) row.push_back( d );
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

//======================================================================
int main(int argc, char* argv[])
{
    
    CommandLineParser parser(argc, argv,
                             "{@AKAZE | kpts1_Akaze.txt | Input Akaze Matches}"
                             "{@ORB | kpts1_ORB.txt | Input ORB Matches}"
                             "{@SIFT | kpts1_SIFT.txt | Input SIFT Matches}"
                             "{@SURF | kpts1_SURF.txt | Input SURF Matches}");
    
    auto data = loadtxt<TYPE>( parser.get<String>("@AKAZE").c_str() );
    auto akaze_end = data.size();
    auto temp = loadtxt<TYPE>( parser.get<String>("@ORB").c_str() );
    concat(data,temp);
    auto orb_end = data.size();
    temp = loadtxt<TYPE>( parser.get<String>("@SIFT").c_str() );
    concat(data,temp);
    auto sift_end = data.size();
    temp = loadtxt<TYPE>( parser.get<String>("@SURF").c_str() );
    concat(data,temp);
    auto surf_end = data.size();
    
    for (decltype(akaze_end) akaze_idx = 0; akaze_idx < akaze_end; akaze_idx++) {
        auto x_akaze = data[akaze_idx][0], y_akaze = data[akaze_idx][1];
        for (decltype(akaze_end) orb_idx = akaze_end; orb_idx < orb_end; orb_idx++) {
            auto   x_orb = data[orb_idx][0],     y_orb = data[orb_idx][1];
            if (x_akaze == x_orb && y_akaze == y_orb) {
                if (data[akaze_idx].size() == 2)
                    data[akaze_idx].push_back(2);
                else {
                    auto old = data[akaze_idx].back();
                    data[akaze_idx].pop_back();
                    data[akaze_idx].push_back( old + 1 );
                }
                break;
            }
        }
        for (decltype(orb_end) sift_idx = orb_end; sift_idx < sift_end; sift_idx++) {
            auto   x_sift = data[sift_idx][0],     y_sift = data[sift_idx][1];
            if (x_akaze == x_sift && y_akaze == y_sift) {
                if (data[akaze_idx].size() == 2)
                    data[akaze_idx].push_back(2);
                else {
                    auto old = data[akaze_idx].back();
                    data[akaze_idx].pop_back();
                    data[akaze_idx].push_back( old + 1 );
                }
                break;
            }
        }
        for (decltype(akaze_end) surf_idx = sift_end; surf_idx < surf_end; surf_idx++) {
            auto   x_surf = data[surf_idx][0],     y_surf = data[surf_idx][1];
            if (x_akaze == x_surf && y_akaze == y_surf) {
                if (data[akaze_idx].size() == 2)
                    data[akaze_idx].push_back(2);
                else {
                    auto old = data[akaze_idx].back();
                    data[akaze_idx].pop_back();
                    data[akaze_idx].push_back( old + 1 );
                }
                break;
            }
        }
    }
    
    for (decltype(akaze_end) orb_idx = akaze_end; orb_idx < orb_end; orb_idx++) {
        auto x_orb = data[orb_idx][0], y_orb = data[orb_idx][1];
        for (decltype(orb_end) sift_idx = orb_end; sift_idx < sift_end; sift_idx++) {
            auto x_sift = data[sift_idx][0], y_sift = data[sift_idx][1];
            if (x_orb == x_sift && y_orb == y_sift) {
                if (data[orb_idx].size() == 2)
                    data[orb_idx].push_back(2);
                else {
                    auto old = data[orb_idx].back();
                    data[orb_idx].pop_back();
                    data[orb_idx].push_back( old + 1 );
                }
                break;
            }
        }
        for (decltype(sift_end) surf_idx = sift_end; surf_idx < surf_end; surf_idx++) {
            auto   x_surf = data[surf_idx][0],     y_surf = data[surf_idx][1];
            if (x_orb == x_surf && y_orb == y_surf) {
                if (data[orb_idx].size() == 2)
                    data[orb_idx].push_back(2);
                else {
                    auto old = data[orb_idx].back();
                    data[orb_idx].pop_back();
                    data[orb_idx].push_back( old + 1 );
                }
                break;
            }
        }
    }
    
    for (decltype(orb_end) sift_idx = orb_end; sift_idx < sift_end; sift_idx++) {
        auto x_sift = data[sift_idx][0], y_sift = data[sift_idx][1];
        for (decltype(sift_end) surf_idx = sift_end; surf_idx < surf_end; surf_idx++) {
            auto   x_surf = data[surf_idx][0],     y_surf = data[surf_idx][1];
            if (x_sift == x_surf && y_sift == y_surf) {
                if (data[sift_idx].size() == 2)
                    data[sift_idx].push_back(2);
                else {
                    auto old = data[sift_idx].back();
                    data[sift_idx].pop_back();
                    data[sift_idx].push_back( old + 1 );
                }
                break;
            }
        }
    }
    
    ofstream out( "saida.txt" );
    print( data , out );
    out.close();
}

