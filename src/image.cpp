/* Tests for final version of Luiz Otávio's TCC */

#include <algorithm>
#include <random>

#include "image.hpp"
#include "point.hpp"
#include "macros.hpp"

namespace lo {

Image::Image(size_t idx, std::string path) : index(idx), filename(path) {
}

bool PointCompair::operator ()(cv::Point2f const& a, cv::Point2f const& b) {
    return (a.x < b.x) || (a.x == b.x && a.y < b.y);
}

bool Image::computeAndDetect(const cv::Ptr<cv::Feature2D> &detector,
                             std::string roi,
                             size_t scale,
                             bool limitkpts,
                             size_t nfeatures,
                             bool verbose) {
    // Try to read the image
    cv::Mat img;
    t_read = GET_F_TIME(img, cv::imread(filename, cv::IMREAD_GRAYSCALE));
    // Check if image is loaded
    if (img.data != NULL) {
        // Aplly scale, if is needed
        if (scale > 1) {
            cv::Mat shrink;
            cv::resize(img, shrink, cv::Size(), 1.0/scale, 1.0/scale, cv::INTER_AREA);
            img = shrink.clone();
        }
        // Define gruber region of interest to get keypoints
        cv::Mat mask;
        if (!roi.empty()) {
            cv::Mat originMask = cv::imread(roi, cv::IMREAD_GRAYSCALE);
            if (originMask.data == NULL) {
                std::cerr << "Failed to open file: " << roi << std::endl;
                return false;
            }
            cv::resize(originMask, mask, img.size(), 0, 0, cv::INTER_LINEAR);
        }
        // Detect keypoints
        t_detect   = GET_P_TIME(detector->detect(img, keypoints, mask));
        // Apply restrictions, if is necessary
        if (limitkpts && nfeatures < keypoints.size()) {
            // This implies on points shuffle to ensure normal distribution
            auto rng = std::default_random_engine {};
            std::shuffle(keypoints.begin(), keypoints.end(), rng);
            // And vector clipping
            keypoints.erase(keypoints.begin()+nfeatures, keypoints.end());
        }
        // Compute descriptors to keypoints
        t_descript = GET_P_TIME(detector->compute(img, keypoints, descriptors ));
        // And compute memory usage to main image objects
        m_read = GET_CVMAT_USAGE( img );
        m_detect = GET_VECTOR_USAGE( keypoints );
        m_descript = GET_CVMAT_USAGE( descriptors );
    }
    // Abort the process if there are any exceptions
    else {
        std::cerr << "Could not read the file " << filename << std::endl;
        return false;
    }
    // Reports the image's keypoint count when prompted
    if (verbose) {
        std::cout << "Image " << filename << " processing...\n";
        std::cout << "Time Opening Image: " << t_read.count() << "\n";
        std::cout << "Memory usage on reading image: " << HUMAN_READABLE(m_read) << "\n";
        std::cout << "Number of keypoints: " << keypoints.size() << "\n";
        std::cout << "Time on keypoints detection: " << t_detect.count() << "\n";
        std::cout << "Memory usage on keypoints detection: " << HUMAN_READABLE(m_detect) << "\n";
        std::cout << "Time on keypoints description: " << t_descript.count() << "\n";
        std::cout << "Memory usage on keypoints description: " << HUMAN_READABLE(m_descript) << "\n"<< "\n";
    }
    // Abort if the number of keypoints is insufficient for the other processes in the flow
    if (keypoints.size() < 4) {
        std::cerr << "The process failed to define keypoints on the set of images!\n";
        return false;
    }
    return true;
}

}
