/* Tests for final version of Luiz Otávio's TCC */
#ifndef LO_POINT_H
#define LO_POINT_H



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



// Point classes definition
namespace lo {

class Measure {
public:
    // Attributes
    size_t index;
    cv::Point2f pt;

    // Constructor
    Measure(size_t index, cv::Point2f pt);
};

class Point {
public:
    // Attributes
    size_t index;
    std::vector< Measure > measures;

    // Constructor
    Point(size_t index = 0);

    // Methods
    bool add(const Measure &measure);
};

}

#endif
