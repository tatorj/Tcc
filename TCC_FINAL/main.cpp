/* Tests for final version of Luiz Otávio's TCC */


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


// Metric macros
#define GET_P_TIME(p) ({\
    auto t0 = std::chrono::high_resolution_clock::now();\
    p;\
    auto t1 = std::chrono::high_resolution_clock::now();\
    t1 - t0; })

#define GET_F_TIME(r, f) ({\
    auto t0 = std::chrono::high_resolution_clock::now();\
    r = f;\
    auto t1 = std::chrono::high_resolution_clock::now();\
    t1 - t0; })

#define GET_VECTOR_USAGE(vec) ({\
    vec.size() * sizeof(vec.front()); })

#define GET_2DVECTOR_USAGE(vec) ({\
    vec.size() * vec.front().size() * sizeof(vec.front().front()); })

#define GET_CVMAT_USAGE(mat) ({\
    (mat.cols*mat.rows) * sizeof(mat.type()); })

#define HUMAN_READABLE(bytes) ({\
	std::string suffix[] = {"B", "KB", "MB", "GB", "TB"};\
	char length = sizeof(suffix) / sizeof(suffix[0]);\
	double dblBytes = bytes;\
	unsigned char i = 0;\
	if (bytes > 1024)\
		for (; bytes / 1024 > 0 && i < length - 1; i++, bytes /= 1024)\
			dblBytes = bytes / 1024.0;\
    std::string result = std::to_string(dblBytes);\
    result.substr(0, result.size()-4) + " " + suffix[i]; })



// Classes definition
namespace lo {

    class Measure {
        public:
            // Attributes
            size_t index;
            cv::Point2f pt;

            // Constructor
            Measure(size_t index, cv::Point2f pt)
                {this->index = index; this->pt = pt;}
    };

    class Point {
        public:
            // Attributes
            size_t index;
            std::vector< Measure > measures;

            // Contrusctor
            Point(size_t index = 0)
                {this->index = index;}

            // Methods
            bool add(const Measure &measure);
    };

    class PointCompair {
        public:
            // Methods
            bool operator ()(cv::Point2f const& a, cv::Point2f const& b)
                { return (a.x < b.x) || (a.x == b.x && a.y < b.y); }
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
			Image(size_t idx = 0, std::string path = "") : index(idx), filename(path)
				{}

			// Methods
			bool computeAndDetect(const cv::Ptr<cv::Feature2D> &detector, bool verbose = false);
			Image& operator =(const Image &obj)
				{this->index = obj.index; this->filename = obj.filename; return *this;}
	};

	class Pair {
        public:
            // Attributes
            cv::Mat homography;
            double RMSE;
            std::vector< cv::DMatch > matches;
            Image *left, *right;
            bool discarded;
            std::chrono::duration<double, std::milli> t_match, t_correct;
            size_t m_match, m_correct;

            // Constructor
            Pair(Image *left = nullptr, Image *right = nullptr)
                { this->left = left; this->right = right;
                  discarded = true; m_match = m_correct = 0; }

            // Methods
            bool checkHomography(const cv::Ptr<cv::DescriptorMatcher> &matcher, double maximumError = 2.0, bool crosscheck = false, bool verbose = false);
	};

	class ProcessController {
            // Command Line arguments
            bool verbose;
            bool crosscheck;
            double residue;
            size_t startPointIndex;
            std::string detectorType;
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



// Main routine
int main (int argc, char** argv) {
	lo::ProcessController controller;

	if ( controller.readArguments(argc, argv) && controller.runProcesses() )
        controller.saveResults();
    else
        controller.printUsage();

	return 0;
}



// Method's implementation
namespace lo {

    static inline std::string &ltrim(std::string &s) {
        // Left trim string to avoid some errors on read files
        // from https://stackoverflow.com/questions/216823/how-to-trim-a-stdstring
        s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
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

	bool Image::computeAndDetect(const cv::Ptr<cv::Feature2D> &detector, bool verbose) {
        // Try to read the image
		cv::Mat img;
		t_read = GET_F_TIME(img, cv::imread(filename, cv::IMREAD_GRAYSCALE));
		// Check if image is loaded
		if (img.data != NULL) {
            // Detect and compute descriptors to key points
			t_detect   = GET_P_TIME(detector->detect(img, keypoints));
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
            std::cout << "Number of keypoints: " << keypoints.size() << "\n";
        }
		return true;
	}

    bool Pair::checkHomography(const cv::Ptr<cv::DescriptorMatcher> &matcher, double me, bool crosscheck, bool verbose) {
        // Running the images matching
        std::vector< std::vector< cv::DMatch > > allMatches;
        std::vector< cv::DMatch > goodMatches;
        if (crosscheck) {
            t_match = GET_P_TIME(matcher->knnMatch(left->descriptors, right->descriptors, allMatches,1));
            for(size_t i = 0; i < allMatches.size(); i++) {
                cv::DMatch first = allMatches[i][0];
                if(first.distance != 0 || (first.queryIdx > 0 && first.queryIdx < (int)left->keypoints.size()))
                    goodMatches.push_back(first);
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
                // Save the inlier match
                matches.push_back(goodMatches[i]);
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
                RMSE += pow(diff.at<double>(0), 2) + pow(diff.at<double>(1), 2);
            }
        }
        // Obtain the square root of residuals by the number of solution points
        RMSE = sqrt(RMSE/N);

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
            std::cout << "Good ratio matches: " << goodMatches.size() << "\n";
            std::cout << "Inlier matches: " << N << "\n";
            std::cout << "The homography matrix is:\n" << homography << "\n";
            std::cout << "RMSE: " << RMSE << "\n";
        }
        return true;
    }

    bool ProcessController::readArguments(int argc, char **argv) {
        const std::string keys =
              "{@imagelist |<none>|list with index and path for images to process}"
              "{@pairslist |<none>|list with index of images in pairs to process}"
              "{@resultname|<none>|filename to save the resulting image measurements process}"
              "{@pointsname|      |filename to save point's indexes and types as needed by e-foto}"
              "{detector  d|AKAZE |select detector type between AKAZE, ORB, SIFT or SURF}"
              "{mode      m|FILE  |select mode of pair aquisition between SEQUENCE, ALL or FILE guided}" // TODO: tratar estes modos
              "{sindex    s|1     |start index to new stich points}"
              "{residue   r|2.0   |define maximum residue for geometric solution}"
              "{ccheck    c|      |use crosscheck with brute force matcher (default is use flann with Lowe's ratio test)}"
              "{verbose   v|      |show all internal process messages}"
              "{help      h|      |show help message}";
        parser = cv::makePtr<cv::CommandLineParser>(argc, argv, keys);

        if ( parser->has("help") )
            return false;

        verbose = parser->has("verbose");
        crosscheck = parser->has("ccheck");

        imagelist = parser->get<std::string>("@imagelist");
        pairslist = parser->get<std::string>("@pairslist");
        resultname = parser->get<std::string>("@resultname");
        pointsname = parser->get<std::string>("@pointsname");

        // TODO: definir mais argumentos e programar a decodificação destes.
        // Exemplos incluem:
        // - limitar o número de pontos na saída
        // - limitar o número de mínimo de medidas esperadas
        // - limiares de corte para homografia
        // - definir recorte de imagens // <+++
        // - definir redimensionamento de imagens
        // - listar apenas os pares (e homografias destes)
        // - processar todos os pares possíveis (ignorar entrada de pares)
        // - processar pares em sequencia (ignorar entrada de pares)
        // - Usar máscara binária (gruber) ao detectar pontos chave
        // - Limitar número de pontos chaves (especialmente útil para o orb)
        detectorType = parser->get<std::string>("detector");
        startPointIndex = parser->get<size_t>("sindex");
        residue = parser->get<double>("residue");
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

        if (detectorType == "AKAZE") {
            detector = cv::AKAZE::create();
        }
        else if (detectorType == "ORB") {
            detector = cv::ORB::create(100000);
        }
        else if (detectorType == "SIFT") {
            detector = cv::SIFT::create();
        }
        else if (detectorType == "SURF") {
            detector = cv::xfeatures2d::SURF::create();
        }
        else {
            std::cerr << "Unexpected detector type!\n";
        }

        if (crosscheck) {
            if (detectorType == "AKAZE" || detectorType == "ORB")
                matcher = cv::makePtr<cv::BFMatcher>(cv::NORM_HAMMING, crosscheck);
            else
                matcher = cv::makePtr<cv::BFMatcher>(cv::NORM_L2, crosscheck);
        }
        else {
            // TODO: Determinar melhor os parametros mais adequados para o LSH
            if (detectorType == "AKAZE" || detectorType == "ORB")
                matcher = cv::makePtr<cv::FlannBasedMatcher>( cv::makePtr<cv::flann::LshIndexParams>(3,20,2) );
            else
                matcher = cv::makePtr<cv::FlannBasedMatcher>( );
        }
        detector->descriptorType();

        // Set the initial state on counters
        stich_creation = stich_update = stich_merge = 0;
        return true;
    }

    bool ProcessController::runProcesses() {
        // Try to read address list from images and instantiate related objects
        std::ifstream inputImages(imagelist);
        if (inputImages.is_open()) {
            size_t index;
            std::string filename;
            while (inputImages >> index) {
                if (images.find(index) == images.end()) {
                    std::getline(inputImages,filename);
                    images[index] = Image(index, ltrim(filename));
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
        // Try to read pairs list from images and instantiate related objects
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

        // Image processing may abort if any image cannot be opened
        for (auto &indexed_image: images) {
            if (!indexed_image.second.computeAndDetect( this->detector, verbose ))
                return false;
        }

        // Pair processing does not abort execution, but pairs can be discarded
        for (size_t i = 0; i < pairs.size(); i++)
            pairs[i].checkHomography( this->matcher, residue, crosscheck, verbose );

        // Make the measurement selection process
        t_stich = GET_P_TIME(makePointList(verbose));

        // Report main parts time consumption when prompted
        if (verbose) {
            // Make zero allocated durations
            std::chrono::duration<double, std::milli> read, detect, descript, match, correct;
            read = detect = descript = match = correct = std::chrono::duration<double, std::milli>::zero();
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
            std::cout << "Total time reading images: " << read.count() << "\n";
            std::cout << "Total time on keypoints detection: " << detect.count() << "\n";
            std::cout << "Total time on keypoints description: " << descript.count() << "\n";
            std::cout << "Total time on matching keypoints: " << match.count() << "\n";
            std::cout << "Total time on geometry verification: " << correct.count() << "\n";
            std::cout << "Total time on stich point registration: " << t_stich.count() << "\n";

            std::cout << "Total of stiches created: " << stich_creation << "\n";
            std::cout << "Total of stiches updated: " << stich_update << "\n";
            std::cout << "Total of stiches merged: "  << stich_merge << "\n";

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
        // Save the main results, all digital images measurements
        std::ofstream imageMeasures(resultname);
        if (imageMeasures.is_open()) {
            for (auto point = points.begin(); point != points.end(); point++)
            {
                if (point->index != 0)
                {
                    if (point != points.begin())
                        imageMeasures << std::endl;
                    size_t n = point->measures.size();
                    for (size_t i = 0; i < n; i++)
                        imageMeasures << point->measures[i].index << "\t"
                                      << point->index << "\t"
                                      << point->measures[i].pt.x << "\t"
                                      << point->measures[i].pt.y << ((i == n-1)?"":"\n");
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
                for (auto point = points.begin(); point != points.end(); point++)
                {
                    if (point->index != 0)
                    {
                        if (point != points.begin())
                            pointENH << std::endl;
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
                size_t f_match = pairs[i].matches[j].queryIdx;
                size_t s_match = pairs[i].matches[j].trainIdx;
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
            if (point->measures.size() > 0) {
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
