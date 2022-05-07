/* Tests for final version of Luiz Otávio's TCC */
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
#include <opencv2/imgproc.hpp>
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
