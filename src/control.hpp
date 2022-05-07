/* Tests for final version of Luiz Otávio's TCC */
#ifndef LO_CONTROL_H
#define LO_CONTROL_H



// STL dependencies
#include <map>
#include <list>
#include <vector>
#include <string>
#include <chrono>
#include <utility>
#include <fstream>
#include <iostream>

/* Notes on the adoption of STL containers in this project:
 * We prefer to use vectors, but this project requires at least one list, as
 * this structure guarantees that the iterators (or even simple pointers) will
 * not be invalidated when a new element is added or even removed. Maps were
 * also used to ensure a quick search for measurements taken on images and
 * images by index.
 */



// OpenCV 4.5.4 dependencies
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#ifdef NONFREEAVAILABLE
#include <opencv2/xfeatures2d.hpp>
#endif

/* Notes on the adoption of openCV algorithms:
 * Some external and non-free opencv classes used for academic tests are removed
 * from compilation by default if opencv from the standard linux repository is
 * in use.
 */



// Join all project classes
#include "image.hpp"
#include "pair.hpp"
#include "point.hpp"



// ProcessControl class definition
namespace lo {

class ProcessController {
    // Command Line arguments
    bool verbose;
    bool crosscheck;
    bool limitkpts;
    bool scapePointList;
    double residue;
    size_t nfeatures;
    size_t imageScale;
    size_t startPointIndex;
    size_t limitMatches;
    size_t nMeasures;
    std::string detectorType;
    std::string roiFile;
    std::string mode;
    std::string imagelist, pairslist, resultname, pointsname;

    // Internal method
    void makePointList(bool verbose = false);

public:
    // Attributes
    std::list< Point > points;
    std::map< size_t, Image > images;
    std::vector< Pair > pairs;
    cv::Ptr<cv::Feature2D> detector;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    cv::Ptr<cv::CommandLineParser> parser;
    std::chrono::duration<double, std::milli> t_stich;
    size_t m_stich, stich_creation, stich_update, stich_merge;

    // Public methods
    bool readArguments(int argc, char **argv);
    bool runProcesses();
    bool saveResults();
    void printUsage();
};

}

#endif
