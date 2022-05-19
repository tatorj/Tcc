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

#ifndef LO_PAIR_H
#define LO_PAIR_H

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

// Pair class definition
namespace lo {

class Image;

class Pair {
public:
    // Attributes
    Image *left, *right;
    bool discarded;
    cv::Mat homography;
    double RMSE;
    std::vector< double > errors;
    std::vector< std::pair< double, cv::DMatch > > matches;
    std::chrono::duration<double, std::milli> t_match, t_correct;
    size_t m_match, m_correct;

    // Constructor
    Pair(Image *left = nullptr, Image *right = nullptr);

    // Methods
    bool checkHomography(const cv::Ptr<cv::DescriptorMatcher> &matcher,
                         double maximumError = 2.0,
                         size_t limit = 0,
                         bool crosscheck = false,
                         double rate = 0.0,
                         bool verbose = false);
};

}

#endif
