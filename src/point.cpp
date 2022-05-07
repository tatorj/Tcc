/* Tests for final version of Luiz Otávio's TCC */

#include "point.hpp"

namespace lo {

Measure::Measure(size_t index, cv::Point2f pt) {
    this->index = index;
    this->pt = pt;
}

Point::Point(size_t index) {
    this->index = index;
}

bool Point::add(const Measure &measure) {
    // Prevents different measurements of the same point for an image
    for (size_t i = 0; i < measures.size(); i++)
        if (measures[i].index == measure.index)
            return false;
    // And add the measurement if is relevant
    measures.push_back(measure);
    return true;
}

}
