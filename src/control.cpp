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

#include "control.hpp"
#include "macros.hpp"

namespace lo {

static inline std::string &ltrim(std::string &s) {
    // Left trim string to avoid some errors on read files
    // from https://stackoverflow.com/questions/216823/how-to-trim-a-stdstring
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                    std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

bool ProcessController::readArguments(int argc, char **argv) {

    const std::string keys =
            "{@imagelist  |<none>|list with index and path for images to process}"
            "{@pairslist  |<none>|list with index of images in pairs to process}"
            "{@resultname |<none>|filename to save the resulting image measurements process}"
            "{@pointsname |      |filename to save point's indexes and types as needed by e-foto}"
            "{crosscheck c|      |use crosscheck with brute force matcher (default is use flann with Lowe's ratio test)}"
            "{detector   d|ORB   |select detector type between AKAZE, ORB, SIFT or SURF}"
            "{number_f   f|10000 |suggest a number (f) of features to retain (adopted by ORB)}"
            "{force_f    F|      |forces the upper bound (f) of features to retain on detection phase}"
            "{gruber_roi g|      |sets a path to gruber's region of interest image file}"
            "{init_index i|1     |start index to new stich points}"
            "{help       h|      |show help message}"
            "{mode       m|FILE  |select mode of pair aquisition between SEQUENCE, ALL or FILE guided}"
            "{n_measures n|0     |filter points by a minimum (n) of image measurements}"
            "{limit_out  o|0     |limit size of measurements per pair on the output}"
            "{list_pairs l|0.0   |write a list of pairs, instead of list of measures, keeping those that can have a geometric solution with an inlier rate (l)}"
            "{residue    r|2.0   |define maximum residue for geometric solution}"
            "{scale      s|1     |a scale denominator to reduce the images size}"
            "{verbose    v|      |show all internal process messages}";
    parser = cv::makePtr<cv::CommandLineParser>(argc, argv, keys);

    if ( parser->has("help") )
        return false;

    verbose = parser->has("verbose");
    crosscheck = parser->has("crosscheck");
    limitkpts = parser->has("force_f");

    mode = parser->get<std::string>("mode");
    imagelist = parser->get<std::string>("@imagelist");
    pairslist = parser->get<std::string>("@pairslist");
    resultname = parser->get<std::string>("@resultname");
    pointsname = parser->get<std::string>("@pointsname");
    detectorType = parser->get<std::string>("detector");
    roiFile = parser->get<std::string>("gruber_roi");
    startPointIndex = parser->get<size_t>("init_index");
    nMeasures = parser->get<size_t>("n_measures");
    limitMatches = parser->get<size_t>("limit_out");
    imageScale = parser->get<size_t>("scale");
    nfeatures = parser->get<int>("number_f");
    residue = parser->get<double>("residue");
    inlierRate = parser->get<double>("list_pairs");
    scapePointList = (inlierRate > 0.0) && (inlierRate <= 1.0);
    if (!parser->check())
    {
        parser->printErrors();
        return false;
    }

    // Avoiding use SURF algorithm when nonfree definition is not available.
#ifndef NONFREEAVAILABLE
    if (detectorType == "SURF") {
        std::cerr << "This detector requires availability of nonfree opencv!\n";
        return false;
    }
#endif

    // Configure detector
    if (detectorType == "AKAZE") {
        detector = cv::AKAZE::create();
    }
    else if (detectorType == "ORB") {
        detector = cv::ORB::create(nfeatures);
    }
    else if (detectorType == "SIFT") {
        detector = cv::SIFT::create();
    }
    else if (detectorType == "SURF") {
        detector = cv::xfeatures2d::SURF::create();
    }
    else {
        std::cerr << "Unexpected detector type!\n";
        return false;
    }

    // Configure matcher
    if (crosscheck) {
        if (detectorType == "AKAZE" || detectorType == "ORB")
            matcher = cv::makePtr<cv::BFMatcher>(cv::NORM_HAMMING, crosscheck);
        else
            matcher = cv::makePtr<cv::BFMatcher>(cv::NORM_L2, crosscheck);
    }
    else {
        // @TODO: Find better guidance on how lsh index parameters can be varied as data volume increases
        if (detectorType == "AKAZE" || detectorType == "ORB")
            matcher = cv::makePtr<cv::FlannBasedMatcher>( cv::makePtr<cv::flann::LshIndexParams>(1,20,2) );
        else
            matcher = cv::makePtr<cv::FlannBasedMatcher>( );
    }
    detector->descriptorType();

    // Set the initial state on counters
    stich_creation = stich_update = stich_merge = 0;
    return true;
}

bool ProcessController::runProcesses() {
    std::chrono::duration<double, std::milli> zero;
    zero = std::chrono::duration<double, std::milli>::zero();

    // Try to read address list from images and instantiate related objects
    std::ifstream inputImages(imagelist);
    if (inputImages.is_open()) {
        size_t index, last_index;
        std::string filename;
        bool sequence = mode == "SEQUENCE";
        while (inputImages >> index) {
            if (images.find(index) == images.end()) {
                std::getline(inputImages,filename);
                images[index] = Image(index, ltrim(filename));
                // Make pairs with image's list sequence when defined by user
                if (sequence) {
                    if (images.size() > 1) {
                        pairs.push_back( Pair(&images[last_index], &images[index]) );
                    }
                    last_index = index;
                }
            }
            // Avoid index colision
            else {
                std::cerr << "There are repeated image indexes in the list!\n";
                inputImages.close();
                return false;
            }
        }
    }
    else {
        std::cerr << "Failed to open file: " << imagelist << std::endl;
        return false;
    }

    // Select pairs as defined by user
    if (mode == "FILE") {
        // FILE mode read image pair's list, create pairs and link to images
        std::ifstream inputPairs(pairslist);
        if (inputPairs.is_open()) {
            size_t first, second;
            while (inputPairs >> first >> second) {
                auto left = images.find(first), right = images.find(second);
                if (left != images.end() && right != images.end())
                    pairs.push_back( Pair(&left->second, &right->second) );
                // Avoid loose index
                else {
                    std::cerr << "There are loosed image index on the list!\n";
                    inputImages.close();
                    return false;
                }
            }
        }
        else {
            std::cerr << "Failed to open file: " << pairslist << std::endl;
            return false;
        }
    }
    else if (mode == "ALL") {
        // ALL mode define all pairs
        for (auto &pivot: images)
            for (auto &elem: images)
                if (elem.first > pivot.first)
                    pairs.push_back( Pair(&images[pivot.first], &images[elem.first]) );
    }
    else if (mode != "SEQUENCE") {
        // SEQUENCE mode is handled while reading the image list
        // so it remains to check if an unexpected mode has been passed
        std::cerr << "Unexpected mode has passed!\n";
        return false;
    }

    // Image processing may abort if any image cannot be opened
    for (auto &indexed_image: images) {
        if (!indexed_image.second.computeAndDetect( this->detector,
                                                    roiFile,
                                                    imageScale,
                                                    limitkpts,
                                                    nfeatures,
                                                    verbose ))
            return false;
    }

    // Pair processing does not abort execution, but pairs can be discarded
    for (size_t i = 0; i < pairs.size(); i++)
        pairs[i].checkHomography( this->matcher,
                                  residue,
                                  limitMatches,
                                  crosscheck,
                                  inlierRate,
                                  verbose );

    // Make the measurement selection process
    if (scapePointList) {
        t_stich = zero;
        m_stich = 0;
    }
    else
        t_stich = GET_P_TIME(makePointList(verbose));

    // Report main parts time consumption when prompted
    if (verbose) {
        // Make zero allocated durations
        std::chrono::duration<double, std::milli> read, detect, descript, match, correct;
        read = detect = descript = match = zero;
        size_t m_read = 0, m_detect = 0, m_descript = 0, m_match = 0, m_correct = 0;
        // Sum ellapsed time
        for (auto &indexed_image: images) {
            auto image = indexed_image.second;
            read += image.t_read;
            detect += image.t_detect;
            descript += image.t_descript;
            m_read += image.m_read;
            m_detect += image.m_detect;
            m_descript += image.m_descript;
        }
        for (size_t i = 0; i < pairs.size(); i++) {
            match += pairs[i].t_match;
            correct += pairs[i].t_correct;
            m_match += pairs[i].m_match;
            m_correct += pairs[i].m_correct;
        }
        // And report it
        std::cout << "Stiches created: " << stich_creation << "\n";
        std::cout << "Stiches updated: " << stich_update << "\n";
        std::cout << "Stiches merged: "  << stich_merge << "\n" << "\n";

        std::cout << "Total time reading images: " << read.count() << "\n";
        std::cout << "Total time on keypoints detection: " << detect.count() << "\n";
        std::cout << "Total time on keypoints description: " << descript.count() << "\n";
        std::cout << "Total time on matching keypoints: " << match.count() << "\n";
        std::cout << "Total time on geometry verification: " << correct.count() << "\n";
        std::cout << "Total time on stich point registration: " << t_stich.count() << "\n";

        std::cout << "Total memory usage on reading images: " << HUMAN_READABLE(m_read) << "\n";
        std::cout << "Total memory usage on keypoints detection: " << HUMAN_READABLE(m_detect) << "\n";
        std::cout << "Total memory usage on keypoints description: " << HUMAN_READABLE(m_descript) << "\n";
        std::cout << "Total memory usage on correlate descriptors: " << HUMAN_READABLE(m_match) << "\n";
        std::cout << "Total memory usage on remaining matches: " << HUMAN_READABLE(m_correct) << "\n";
        std::cout << "Total memory usage on stich point registration: " << HUMAN_READABLE(m_stich) << "\n";
    }
    return true;
}

bool ProcessController::saveResults() {
    // Write a list of pairs and scape when requested by the user
    if (scapePointList) {
        std::ofstream pairsList(resultname);
        if (pairsList.is_open()) {
            bool flag = false;
            for (size_t i = 0; i < pairs.size(); i++) {
                if (!pairs[i].discarded) {
                    if (flag)
                        pairsList << std::endl;
                    else
                        flag = true;
                    pairsList << pairs[i].left->index << "\t" << pairs[i].right->index;
                }
            }
            pairsList.close();
            return true;
        }
        else {
            std::cerr << "Failed to create file: " << resultname << std::endl;
            return false;
        }
    }

    // Save the main results, all digital images measurements
    std::ofstream imageMeasures(resultname);
    if (imageMeasures.is_open()) {
        bool flag = false;
        for (auto point = points.begin(); point != points.end(); point++) {
            if (point->index != 0) {
                if (flag)
                    imageMeasures << std::endl;
                else
                    flag = true;
                size_t n = point->measures.size();
                for (size_t i = 0; i < n; i++)
                    imageMeasures << point->measures[i].index << "\t"
                                  << point->index << "\t"
                                  << point->measures[i].pt.x * imageScale << "\t"
                                  << point->measures[i].pt.y * imageScale << ((i == n-1)?"":"\n");
            }
        }
        imageMeasures.close();
    }
    else {
        std::cerr << "Failed to create file: " << resultname << std::endl;
        return false;
    }
    // And e-foto's ENH points, if is needed
    if (!pointsname.empty()) {
        std::ofstream pointENH(pointsname);
        if (pointENH.is_open()) {
            bool flag = false;
            for (auto point = points.begin(); point != points.end(); point++) {
                if (point->index != 0) {
                    if (flag)
                        pointENH << std::endl;
                    else
                        flag = true;
                    pointENH << point->index << "\t" << "Photogrammetric" << "\t"
                             << "0" << "\t" << "0" << "\t" << "0" << "\t"
                             << "0" << "\t" << "0" << "\t" << "0";
                }
            }
            pointENH.close();
        }
        else {
            std::cerr << "Failed to create file: " << pointsname << std::endl;
            return false;
        }
    }

    return true;
}

void ProcessController::printUsage() {
    parser->printMessage();
}

void ProcessController::makePointList(bool verbose) {
    // For each remaining pair
    for (size_t i = 0; i < pairs.size(); i++) {
        if (pairs[i].discarded)
            continue;
        // Get image pointers to the pair in the process
        auto f_image = pairs[i].left;
        auto s_image = pairs[i].right;
        // For each remaining (inlier) match
        for (size_t j = 0; j < pairs[i].matches.size(); j++) {
            // Get the match indexes
            size_t f_match = pairs[i].matches[j].second.queryIdx;
            size_t s_match = pairs[i].matches[j].second.trainIdx;
            // Look on image's keypoints to get matched points
            cv::Point2f f_point = f_image->keypoints[f_match].pt;
            cv::Point2f s_point = s_image->keypoints[s_match].pt;
            // Access image's pointmaps to check for existing measurements
            auto f_pt_id = pairs[i].left->pointmap.find(f_point);
            auto s_pt_id = pairs[i].right->pointmap.find(s_point);
            bool f_isnew = ( f_pt_id == f_image->pointmap.end() );
            bool s_isnew = ( s_pt_id == s_image->pointmap.end() );
            // Chooses between generating, updating or merging stich points
            // based on the state of existence of measurements in the images
            if (f_isnew && s_isnew) {
                // Generate and store point
                stich_creation++;
                Point stich;
                points.push_back(stich);
                // Add measurements
                Measure f_measure(f_image->index, f_point);
                Measure s_measure(s_image->index, s_point);
                points.back().add(f_measure);
                points.back().add(s_measure);
                // Store point and update images
                f_image->pointmap[f_point] = &points.back();
                s_image->pointmap[s_point] = &points.back();
            }
            else if (f_isnew) {
                // Update point to add the first image measurement
                stich_update++;
                Point* stich = s_pt_id->second;
                Measure measure(f_image->index, f_point);
                stich->add(measure);
                // Update the first image
                f_image->pointmap[f_point] = stich;
            }
            else if (s_isnew) {
                // Update point to add the second image measurement
                stich_update++;
                Point* stich = f_pt_id->second;
                Measure measure(s_image->index, s_point);
                stich->add(measure);
                // Update the second image
                s_image->pointmap[s_point] = stich;
            }
            else {
                // Copy all measurements from the second point to the first
                // and update the images that pointed to the second point
                stich_merge++;
                Point* f_stich = f_pt_id->second;
                Point* s_stich = s_pt_id->second;
                for (size_t k = 0; k < s_stich->measures.size(); k++)
                {
                    auto measure = s_stich->measures[k];
                    f_stich->add( measure );
                    s_image->pointmap[ measure.pt ] = f_stich;
                }
                // Avoid the search cost on list to clear the second point
                s_stich->measures.clear();
            }
        }
    }
    // Apply indexes to points
    size_t stichesCount = m_stich = 0;
    for (auto point = points.begin(); point != points.end(); point++) {
        // Avoiding points without measurements
        // or below the number of measurements required, if any
        if ( (nMeasures == 0 && point->measures.size() > 0) ||
             (nMeasures > 0 && point->measures.size() >= nMeasures)) {
            point->index = startPointIndex + stichesCount++;
            // And computing memory usage to main stich point objects
            m_stich += sizeof(point->index) + point->measures.size() * sizeof(Measure);
        }
    }
    // Reports the stich point's count when prompted
    if (verbose) {
        std::cout << "Total stich points: " << stichesCount << "\n";
    }
}

}
