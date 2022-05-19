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
