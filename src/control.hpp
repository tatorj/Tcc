/* Copyright 2022 Luiz Otavio Soares de Oliveira by FEN/UERJ
 * This file is part of the final version of the TCC by Luiz Otavio, as 
 * a requirement for obtaining a degree at this public university under 
 * guidance of Irving Badolato professor.
 * The resulting software is free: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License (GNU GPL) as 
 * published by the Free Software Foundation, either version 3 of the 
 * License, or (at your option) any later version.
 * Our code is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
 * Read the GNU GPL for more details. To obtain a copy of this license 
 * see <http://www.gnu.org/licenses/>.
 */

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

// OpenCV 4.5.4 dependencies
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#ifdef NONFREEAVAILABLE
#include <opencv2/xfeatures2d.hpp>
#endif

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
    double inlierRate;
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
