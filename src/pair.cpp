/* Tests for final version of Luiz Otávio's TCC */

#include <algorithm>

#include "pair.hpp"
#include "image.hpp"
#include "macros.hpp"

namespace lo {

Pair::Pair(Image *left, Image *right) {
    this->left = left;
    this->right = right;
    discarded = true;
    m_match = m_correct = 0;
}

bool compareMatches(const std::pair<double,cv::DMatch> &i,
                    const std::pair<double,cv::DMatch> &j) {
    return i.first < j.first;
}

bool Pair::checkHomography(const cv::Ptr<cv::DescriptorMatcher> &matcher, double me, size_t limit, bool crosscheck, bool verbose) {
    // Running the images matching
    std::vector< std::vector< cv::DMatch > > allMatches;
    std::vector< cv::DMatch > goodMatches;
    if (crosscheck) {
        t_match = GET_P_TIME(matcher->knnMatch(left->descriptors, right->descriptors, allMatches,1));

        // Discard matches based on OpenCV cross-correlation, when requested by the user
        for(size_t i = 0; i < allMatches.size(); i++) {
            if (allMatches[i].size()==1){
                goodMatches.push_back(allMatches[i][0]);
            }
        }
    }
    else {
        t_match = GET_P_TIME(matcher->knnMatch(left->descriptors, right->descriptors, allMatches, 2));

        // Eliminate matches based on the proportion of nearest neighbor
        // distance as an alternative to cross-correlation as described in:
        // www.uio.no/studier/emner/matnat/its/TEK5030/v19/lect/lecture_4_2_feature_matching.pdf
        for(size_t i = 0; i < allMatches.size(); i++) {
            if (allMatches[i].size() == 2) {
                cv::DMatch first = allMatches[i][0];
                float dist1 = allMatches[i][0].distance;
                float dist2 = allMatches[i][1].distance;
                if(dist1 < 0.8 * dist2)
                    goodMatches.push_back(first);
            }
        }
    }

    // Find the homography
    std::vector<cv::Point2f> f_pts, s_pts;
    cv::Mat inliers;
    for( size_t i = 0; i < goodMatches.size(); i++ ) {
        auto match = goodMatches[i];
        auto pair = std::make_pair(match.queryIdx, match.trainIdx);
        f_pts.push_back( left->keypoints[ pair.first ].pt );
        s_pts.push_back( right->keypoints[ pair.second ].pt );
    }
    t_correct = GET_F_TIME(homography, cv::findHomography(f_pts, s_pts, cv::RANSAC, me, inliers));

    // Discard the pair if there is no geometric solution
    size_t N = cv::sum(inliers)[0];
    if (homography.empty() || N < 4) {
        // Report the pair discard when prompted
        if (verbose) {
            std::cout << "No correlation could be found between the pair " << left->index << "x" << right->index << std::endl;
        }
        return false;
    }
    discarded = false;

    // RMSE computing and inlier matches register
    RMSE = 0.0;
    for( size_t i = 0; i < goodMatches.size(); i++ ) {
        if (inliers.at<bool>(i)) {
            // Project the point on the first image onto the second image
            cv::Mat f_point = cv::Mat::ones(3, 1, CV_64F);
            f_point.at<double>(0) = f_pts[i].x;
            f_point.at<double>(1) = f_pts[i].y;
            cv::Mat f_point_projected = homography * f_point;
            f_point_projected /= f_point_projected.at<double>(2);
            // Check the distance between the expected point and the projected one
            cv::Mat s_point = cv::Mat::ones(3, 1, CV_64F);
            s_point.at<double>(0) = s_pts[i].x;
            s_point.at<double>(1) = s_pts[i].y;
            cv::Mat diff = s_point - f_point_projected;
            // Accumulate the residual error values
            double err = pow(diff.at<double>(0), 2) + pow(diff.at<double>(1), 2);
            RMSE += err;
            // Save the inlier match
            matches.push_back( std::make_pair(err,goodMatches[i]));
        }
    }
    // Obtain the square root of residuals by the number of solution points
    RMSE = sqrt(RMSE/N);
    // Sort matches by error
    std::sort(matches.begin(), matches.end(), compareMatches);
    // And crop it if is needed
    if (limit > 0 && matches.size() > limit)
        matches.erase(matches.begin()+limit, matches.end());

    // Compute memory usage to main pair objects
    m_match = GET_2DVECTOR_USAGE( allMatches );
    m_correct = GET_VECTOR_USAGE( matches );

    // Reports the pair's matches count when prompted
    if (verbose) {
        std::cout << "Pair " << left->index << "x" << right->index << " processing...\n";
        if (crosscheck)
            std::cout << "Total distance matches available: " << left->descriptors.rows * right->descriptors.rows << "\n";
        else
            std::cout << "Total distance matches available: " << 2 * allMatches.size() << "\n";
        std::cout << "Good matches: " << goodMatches.size() << "\n";
        std::cout << "Solution (inlier) matches: " << N << "\n";
        std::cout << "The homography matrix is:\n" << homography << "\n";
        std::cout << "RMSE: " << RMSE << "\n";
        std::cout << "Time on matching keypoints: " << t_match.count() << "\n";
        std::cout << "Time on geometry verification: " << t_correct.count() << "\n";
        std::cout << "Memory usage on correlate descriptors: " << HUMAN_READABLE(m_match) << "\n";
        std::cout << "Memory usage on remaining matches: " << HUMAN_READABLE(m_correct) << "\n" << "\n";
    }
    return true;
}

}
