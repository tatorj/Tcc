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

// OpenCV 4.5.X dependencies
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>

/* Notes on the adoption of STL containers in this project:
 * We prefer to use vectors, but this project requires at least one list, as
 * this structure guarantees that the iterators (or even simple pointers) will
 * not be invalidated when a new element is added or even removed. A map was
 * also used to ensure a quick search for measurements taken on images.
 */


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

			// Constructor
			Image(size_t idx, std::string path = "") : index(idx), filename(path)
				{}
			Image(std::string path = "") : filename(path)
				{index = nextIndex();}

			// Methods
			bool computeAndDetect(const cv::Ptr<cv::Feature2D> &detector, bool verbose = false);
			size_t nextIndex() {static size_t idx = 0; return idx++;}
	};

	class Pair {
        public:
            // Attributes
            cv::Mat homography;
            double RMSE;
            std::vector< cv::DMatch > matches;
            Image *left, *right;

            // Constructor
            Pair(Image *left = nullptr, Image *right = nullptr)
                { this->left = left; this->right = right; }

            // Methods
            bool checkHomography(const cv::Ptr<cv::DescriptorMatcher> &matcher, double maximumError = 2.0, bool verbose = false);
	};

	class ProcessController {
        public:
            // Attributes
            bool verbose;
            size_t startPointIndex;
            std::list< Point > points;
            std::vector< Image > images;
            std::vector< Pair > pairs;
            cv::Ptr<cv::Feature2D> detector;
            cv::Ptr<cv::DescriptorMatcher> matcher;

            // Methods
            bool readArguments(int argc, char** argv);
            bool runProcesses();
            bool saveResults();
            void printUsage();
	};
}



// main routine
int main (int argc, char** argv)
{
	lo::ProcessController controller;

	if ( controller.readArguments(argc, argv) && controller.runProcesses() )
        controller.saveResults();
    else
        controller.printUsage();

	return 0;
}



// Method's implementation
namespace lo {
    bool Point::add(const Measure &measure)
    {
        for (size_t i = 0; i < measures.size(); i++)
            if (measures[i].index == measure.index)
                return false;
        measures.push_back(measure);
        return true;
    }

	bool Image::computeAndDetect(const cv::Ptr<cv::Feature2D> &detector, bool verbose) {
        // TODO: incorporar macros de medição (imread, detect and compute)
		cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
		if (img.data == NULL) {
			std::cerr << "Não foi possível abrir a imagem " << filename << std::endl;
			return false;
		}
		else {
			detector->detect(img, keypoints);
			detector->compute(img, keypoints, descriptors );
			if (verbose)
			{
                std::cout << "Image " << filename << " processing...\n";
                std::cout << "Number of keypoints: " << keypoints.size() << "\n";
			}
		}
		return true;
	}

    bool Pair::checkHomography(const cv::Ptr<cv::DescriptorMatcher> &matcher, double me, bool verbose) {
        // TODO: incorporar macros de medição (knnMatch and findHomography)

        // Images matching
        std::vector< std::vector< cv::DMatch > > allMatches;
        matcher->knnMatch(left->descriptors, right->descriptors, allMatches, 2);

        // We eliminate matches based on the proportion of nearest neighbor
        // distance as an alternative to cross-correlation as described in:
        // www.uio.no/studier/emner/matnat/its/TEK5030/v19/lect/lecture_4_2_feature_matching.pdf
        std::vector< cv::DMatch > goodMatches;
        for(size_t i = 0; i < allMatches.size(); i++)
        {
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
        for( size_t i = 0; i < goodMatches.size(); i++ )
        {
            auto match = goodMatches[i];
            auto pair = std::make_pair(match.queryIdx, match.trainIdx);
            f_pts.push_back( left->keypoints[ pair.first ].pt );
            s_pts.push_back( right->keypoints[ pair.second ].pt );
        }
        // TODO: O erro de reprojeção máximo (me) poderia estar entre os argumentos definidos pelo usuário
        homography = cv::findHomography(f_pts, s_pts, cv::RANSAC, me, inliers);

        // TODO: Emitir mensagens de falha caso o processamento de homografia seja impossível
        size_t N = cv::sum(inliers)[0];
        if (homography.empty() || N < 4)
            return false;

        // RMSE computing and inlier matches register
        RMSE = 0.0;
        for( size_t i = 0; i < goodMatches.size(); i++ )
        {
            if (inliers.at<bool>(i))
            {
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

        if (verbose)
        {
            std::cout << "Pair " << left->index << "x" << right->index << " processing...\n";
            std::cout << "Total matches: " << allMatches.size() << "\n";
            std::cout << "Good ratio matches: " << goodMatches.size() << "\n";
            std::cout << "Inlier matches: " << N << "\n";
            std::cout << "The homography matrix is:\n" << homography << "\n";
            std::cout << "RMSE: " << RMSE << "\n";
        }

        // TODO: Justificar na monografia a escolha de RANSAC entre as demais opções
        return true;
    }

    bool ProcessController::readArguments(int argc, char **argv) {
        // TODO: definir os argumentos e programar a decodificação destes.
        startPointIndex = 14;
        verbose = true;

        // TODO: Selecionar detector e descriptors matcher com os argumentos inseridos
        detector = cv::AKAZE::create();
        matcher = new cv::BFMatcher(cv::NORM_HAMMING);
        return true;
    }

    bool ProcessController::runProcesses() {
        // TODO: Revisar o fluxo de atividades, inserir medições, se necessário
        // e gerenciar falhas.

        // TODO: Ler arquivo de controle e instanciar imagens e pares

        // TEST-START
            Image i16(1,"/home/luiz/Documentos/images/16.bmp");
            Image i17(2,"/home/luiz/Documentos/images/17.bmp");
            Image i18(3,"/home/luiz/Documentos/images/18.bmp");
            images.push_back(i16);
            images.push_back(i17);
            images.push_back(i18);
            Pair p1617(&images[0], &images[1]);
            Pair p1718(&images[1], &images[2]);
            pairs.push_back(p1617);
            pairs.push_back(p1718);
        // TEST-END

        // Images processing
        auto dc0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < images.size(); i++)
            images[i].computeAndDetect( this->detector, verbose );
        auto dc1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ddc = dc1 - dc0;
        std::cout << "Processamento de imagens " << ddc.count() << "ms\n";

        // Pairs processing
        auto mt0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < pairs.size(); i++)
            pairs[i].checkHomography( this->matcher, 3.0, verbose );
        auto mt1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dmt = mt1 - mt0;
        std::cout << "Processamento de correlações " << dmt.count() << "ms\n";

        // Points processing
        auto pt0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < pairs.size(); i++)
        {
            // Get image pointers to the pair in the process
            auto f_image = pairs[i].left;
            auto s_image = pairs[i].right;
            for (size_t j = 0; j < pairs[i].matches.size(); j++)
            {
                // Get match indexes
                size_t f_match = pairs[i].matches[j].queryIdx;
                size_t s_match = pairs[i].matches[j].trainIdx;
                // Get matched points
                cv::Point2f f_point = f_image->keypoints[f_match].pt;
                cv::Point2f s_point = s_image->keypoints[s_match].pt;
                // Access images to check for existing measurements
                auto f_pt_id = pairs[i].left->pointmap.find(f_point);
                auto s_pt_id = pairs[i].right->pointmap.find(s_point);
                bool f_isnew = ( f_pt_id == f_image->pointmap.end() );
                bool s_isnew = ( s_pt_id == s_image->pointmap.end() );

                // TEST-START
                if ((!f_isnew) || (!s_isnew))
                {
                    if (!s_isnew) {
                        for (size_t k = 0; k < s_pt_id->second->measures.size(); k++)
                            std::cout << s_pt_id->second->measures[k].pt << std::endl;
                    }
                    std::cout << f_point << "x" << s_point << "\t" << f_isnew << "x" << s_isnew << " " << f_image->index << "x" << s_image->index << "\n";
                }
                // TEST-END

                // Check to generate, update or fusion stitch point
                if (f_isnew && s_isnew)
                {
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
                else if (f_isnew)
                {
                    // Update point to add the first image measurement
                    Point* stich = s_pt_id->second;
                    Measure measure(f_image->index, f_point);
                    stich->add(measure);
                    // Update the first image
                    f_image->pointmap[f_point] = stich;
                }
                else if (s_isnew)
                {
                    // Update point to add the second image measurement
                    Point* stich = f_pt_id->second;
                    Measure measure(s_image->index, s_point);
                    stich->add(measure);
                    // Update the second image
                    s_image->pointmap[s_point] = stich;
                }
                else
                {
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
                    // Avoid the erase cost cleaning the second point
                    s_stich->measures.clear();
                }
            }
        }
        // Apply indexes to points
        size_t stichesCount = 0;
        for (auto point = points.begin(); point != points.end(); point++)
        {
            if (point->measures.size() > 0)
            {
                ++stichesCount;
                point->index = startPointIndex + stichesCount;
            }
        }
        auto pt1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dpt = pt1 - pt0;
        std::cout << "Processamento de medidas " << dpt.count() << "ms\n";


        if (verbose) {
            std::cout << "Total stich points: " << stichesCount << "\n";
        }

        return true;
    }

    bool ProcessController::saveResults() {
        // TODO: revisar o salvamento e garantir o uso de argumentos

        auto sv0 = std::chrono::high_resolution_clock::now();

        // TODO: checar se os arquivos foram abertos corretamente e abortar caso
        // necessário informando o erro

        std::ofstream imageMeasures("final_Matches_AKAZE.txt");
        std::ofstream pointENH("final_ENH_AKAZE.txt");
        for (auto point = points.begin(); point != points.end(); point++)
        {
            if (point->index != 0)
            {
                if (point != points.begin())
                {
                    imageMeasures << std::endl;
                    pointENH << std::endl;
                }
                pointENH << point->index << "\t" << "Photogrammetric" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0";
                size_t n = point->measures.size();
                for (size_t i = 0; i < n; i++)
                    imageMeasures << point->measures[i].index << "\t" << point->index << "\t" << point->measures[i].pt.x << "\t" << point->measures[i].pt.y << ((i == n-1)?"":"\n");
            }
        }
        imageMeasures.close();
        pointENH.close();

        auto sv1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dsv = sv1 - sv0;
        std::cout << "Persistência de dados " << dsv.count() << "ms\n";

        return true;
    }

    void ProcessController::printUsage() {
        // TODO: definir os argumentos e programar a impressão de um manual para
        // uso do programa com comentários para os argumentos disponíveis.
    }

}
