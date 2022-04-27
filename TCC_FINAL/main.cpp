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
 * not be invalidated when a new element is added or even removed. A map was
 * also used to ensure a quick search for measurements taken on images.
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


// Textos para a monografia:

// A ligação entre as classe imagem e ponto faz-se importante em dois momentos e
// nos dois sentidos, sendo caracterizada a ligação com o ponto, isto é, partindo
// da imagem, como prioritária. Isto se justifica pelo algoritmo de integração de
// dos pontos de costura de pares distintos. Nos casos onde uma imagem participa
// de diversos pares é necessária a busca na imagem por pontos a fim de determinar
// se estes já foram registrados durante o processamento de outros pares com a
// mesma imagem. Ao seu tempo, a ligação com a imagem, partindo do ponto, faz-se
// necessária em termos de entrega do produto final, na qual cada medida do ponto
// em distintas imagens deve ser corretamente associada. A associação das medidas
// de cada ponto podem então ser atendidas com uma simples cópia do índice da
// imagem sem que isso implique em perda de performance.

// Para a busca na imagem por ponto adotou-se a estrutura de dados map, da
// biblioteca padrão de C++, pois esta possui tempo de acesso reduzido aplicando
// internamente o algoritmo de busca binária. Outras alternativas seriam adotar
// uma KD-tree (árvore de K dimensões), com tempo de acesso equivalente ao map,
// ou um algoritmo de hash (como implementado em unordered_map) com tempo de
// resposta menor, mas possivelmente com elevado custo de armazenamento.
// Ajustes necessários para uso da classe map com os pontos do opencv foram
// indicados em https://stackoverflow.com/questions/26483306/stdmap-with-cvpoint-as-key.

// PS.: Justificar na monografia a escolha de RANSAC entre as demais opções (LSM, RHO e LMEDS)


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
			Image(size_t idx, std::string path = "") : index(idx), filename(path)
				{}

			// Methods
			bool computeAndDetect(const cv::Ptr<cv::Feature2D> &detector, bool verbose = false);
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
            size_t m_match;

            // Constructor
            Pair(Image *left = nullptr, Image *right = nullptr)
                { this->left = left; this->right = right; discarded = true;}

            // Methods
            bool checkHomography(const cv::Ptr<cv::DescriptorMatcher> &matcher, double maximumError = 2.0, bool verbose = false);
	};

	class ProcessController {
            // Command Line arguments
            bool verbose;
            size_t startPointIndex;
            std::string detectorType;

            // Our measurement selection method
            void makePointList(bool verbose = false);
        public:
            // Attributes
            std::list< Point > points;
            std::vector< Image > images;
            std::vector< Pair > pairs;
            cv::Ptr<cv::Feature2D> detector;
            cv::Ptr<cv::DescriptorMatcher> matcher;
            cv::Ptr<cv::CommandLineParser> parser;

            // Methods
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

    bool Pair::checkHomography(const cv::Ptr<cv::DescriptorMatcher> &matcher, double me, bool verbose) {
        // TODO: computar tamanho das estruturas de correlação
        // Running the images matching
        std::vector< std::vector< cv::DMatch > > allMatches;
        t_match = GET_P_TIME(matcher->knnMatch(left->descriptors, right->descriptors, allMatches, 2));

        // Eliminate matches based on the proportion of nearest neighbor
        // distance as an alternative to cross-correlation as described in:
        // www.uio.no/studier/emner/matnat/its/TEK5030/v19/lect/lecture_4_2_feature_matching.pdf
        std::vector< cv::DMatch > goodMatches;
        for(size_t i = 0; i < allMatches.size(); i++) {
            cv::DMatch first = allMatches[i][0];
            float dist1 = allMatches[i][0].distance;
            float dist2 = allMatches[i][1].distance;
            // TODO: O threshold de boa separação (0.8 para o SIFT) deveria ser ajustado junto com o algoritmo de detecção adotado ou definido pelo usuário
            if(dist1 < 0.8 * dist2)
                goodMatches.push_back(first);
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
        // TODO: O erro de reprojeção máximo (me) poderia estar entre os argumentos definidos pelo usuário
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

        // Reports the pair's matches count when prompted
        if (verbose) {
            std::cout << "Pair " << left->index << "x" << right->index << " processing...\n";
            std::cout << "Total matches: " << allMatches.size() << "\n";
            std::cout << "Good ratio matches: " << goodMatches.size() << "\n";
            std::cout << "Inlier matches: " << N << "\n";
            std::cout << "The homography matrix is:\n" << homography << "\n";
            std::cout << "RMSE: " << RMSE << "\n";
        }
        return true;
    }

    bool ProcessController::readArguments(int argc, char **argv) {
        const std::string keys =
              "{detector d|AKAZE|select detector type between AKAZE, ORB, SIFT or SURF}"
              "{sindex   s|1    |start index to new stich points}"
              "{verbose  v|false|show all internal process messages}"
              "{help     h|false|show help message}";
        parser = new cv::CommandLineParser(argc, argv, keys);

        if ( parser->has("help") && parser->get<bool>("help"))
            return false;

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
        startPointIndex = parser->get<size_t>("sindex");
        verbose = parser->get<bool>("verbose");
        detectorType = parser->get<std::string>("detector");
        if (!parser->check())
        {
            parser->printErrors();
            return false;
        }

        // TODO: Revisar os detectorTypes SIFT e SURF.
#ifndef NONFREEAVAILABLE
        if (detectorType == "SURF") {
            std::cerr << "This detector requires availability of nonfree opencv!\n";
            return false;
        }
#endif
        if (detectorType == "AKAZE") {
            detector = cv::AKAZE::create();
            matcher = new cv::BFMatcher(cv::NORM_HAMMING);
        }
        else if (detectorType == "ORB") {
            detector = cv::ORB::create();
            matcher = new cv::BFMatcher(cv::NORM_HAMMING);
        }
        // else if (detectorType == "SIFT") {
        //     detector = cv::SIFT::create();
        //     matcher = new cv::BFMatcher(cv::NORM_HAMMING);
        // }
        // else if (detectorType == "SURF") {
        //     detector = cv::SURF::create();
        //     matcher = new cv::BFMatcher(cv::NORM_HAMMING);
        // }
        else {
            std::cerr << "Unexpected detector type!\n";
        }

        return true;
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
                    Point* stich = s_pt_id->second;
                    Measure measure(f_image->index, f_point);
                    stich->add(measure);
                    // Update the first image
                    f_image->pointmap[f_point] = stich;
                }
                else if (s_isnew) {
                    // Update point to add the second image measurement
                    Point* stich = f_pt_id->second;
                    Measure measure(s_image->index, s_point);
                    stich->add(measure);
                    // Update the second image
                    s_image->pointmap[s_point] = stich;
                }
                else {
                    // Copy all measurements from the second point to the first
                    // and update the images that pointed to the second point
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
        size_t stichesCount = 0;
        for (auto point = points.begin(); point != points.end(); point++) {
            // Avoiding points without measurements
            if (point->measures.size() > 0) {
                point->index = startPointIndex + stichesCount++;
            }
        }
        // Reports the stich point's count when prompted
        if (verbose) {
            std::cout << "Total stich points: " << stichesCount << "\n";
        }
    }

    bool ProcessController::runProcesses() {
        // TODO: Revisar o fluxo de atividades, inserir medições, se necessário
        // e gerenciar falhas.

        // TODO: Ler arquivo de controle e instanciar imagens e pares

        // TEST-START
            Image i16(1,"../images/16.bmp");
            Image i17(2,"../images/17.bmp");
            Image i18(3,"../images/18.bmp");
            images.push_back(i16);
            images.push_back(i17);
            images.push_back(i18);
            Pair p1617(&images[0], &images[1]);
            Pair p1718(&images[1], &images[2]);
            pairs.push_back(p1617);
            pairs.push_back(p1718);
        // TEST-END

        // Image processing may abort if any image cannot be opened
        for (size_t i = 0; i < images.size(); i++)
            if (!images[i].computeAndDetect( this->detector, verbose ))
                return false;

        // Pair processing does not abort execution, but pairs can be discarded
        for (size_t i = 0; i < pairs.size(); i++)
            pairs[i].checkHomography( this->matcher, 3.0, verbose );

        // Make the measurement selection process
        makePointList(verbose);

        // Report main parts time consumption when prompted
        if (verbose) {
            // TODO: adicionar os tempos de correlação e o consumo de memória
            // Make zero allocated durations
            std::chrono::duration<double, std::milli> read, detect, descript, match, correct;
            read = detect = descript = match = correct = std::chrono::duration<double, std::milli>::zero();
            size_t m_read = 0, m_detect = 0, m_descript = 0;
            // Sum ellapsed time
            for (size_t i = 0; i < images.size(); i++) {
                read += images[i].t_read;
                detect += images[i].t_detect;
                descript += images[i].t_descript;
                m_read += images[i].m_read;
                m_detect += images[i].m_detect;
                m_descript += images[i].m_descript;
            }
            for (size_t i = 0; i < pairs.size(); i++) {
                // TODO: Devemos contar o tempo mesmo de pares descartados?
                match += pairs[i].t_match;
                correct += pairs[i].t_correct;
            }
            // And report it
            std::cout << "Total time reading images: " << read.count() << "\n";
            std::cout << "Total time on keypoints detection: " << detect.count() << "\n";
            std::cout << "Total time on keypoints description: " << descript.count() << "\n";
            std::cout << "Total time on matching keypoints: " << match.count() << "\n";
            std::cout << "Total time on geometry verification: " << correct.count() << "\n";

            std::cout << "Total memory usage on reading images: " << HUMAN_READABLE(m_read) << "\n";
            std::cout << "Total memory usage on keypoints detection: " << HUMAN_READABLE(m_detect) << "\n";
            std::cout << "Total memory usage on keypoints description: " << HUMAN_READABLE(m_descript) << "\n";
        }
        return true;
    }

    bool ProcessController::saveResults() {
        // TODO: revisar o salvamento e garantir o uso de argumentos

        std::ofstream imageMeasures("final_Matches_AKAZE.txt");
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
            std::cerr << "Failed to create file: " << "final_Matches_AKAZE.txt" << std::endl;
            return false;
        }

        std::ofstream pointENH("final_ENH_AKAZE.txt");
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
            std::cerr << "Failed to create file: " << "final_ENH_AKAZE.txt" << std::endl;
            return false;
        }

        return true;
    }

    void ProcessController::printUsage() {
        parser->printMessage();
    }

}
