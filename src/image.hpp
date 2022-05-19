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

#ifndef LO_IMAGE_H
#define LO_IMAGE_H

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
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#ifdef NONFREEAVAILABLE
#include <opencv2/xfeatures2d.hpp>
#endif

// Image classes definition
namespace lo {

class Point;

class PointCompair {
public:
    // Methods
    bool operator ()(cv::Point2f const& a, cv::Point2f const& b);
};

class PointMap: public std::map<cv::Point2f, Point*, PointCompair> {
};

class Image {
public:
    // Attributes
    size_t index;
    std::string filename;
    cv::Mat descriptors;
    std::vector< cv::KeyPoint > keypoints;
    PointMap pointmap;
    std::chrono::duration<double, std::milli> t_read, t_detect, t_descript;
    size_t m_read, m_detect, m_descript;

    // Constructor
    Image(size_t idx = 0,
          std::string path = "");

    // Methods
    bool computeAndDetect(const cv::Ptr<cv::Feature2D> &detector,
                          std::string roi = "",
                          size_t scale = 1,
                          bool limitkpts = false,
                          size_t nfeatures = 10000,
                          bool verbose = false);
};

}

#endif
