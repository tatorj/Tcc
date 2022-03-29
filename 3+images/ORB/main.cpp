#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>      // <- declaramos findHomography
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include <fstream>

using namespace std;
using namespace cv;
const float inlier_threshold = 2.5f; // Distance threshold to identify inliers with homography check
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

int main(int argc, char* argv[]) {
	
	// Iniciação do calculo de tempo
	using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
	auto t1 = high_resolution_clock::now();
	
	// O parâmetro de homografia foi substituído por um limiar
    CommandLineParser parser(argc, argv,
                             "{@img1 | 16.bmp | input image 1}"
                             "{@img2 | 17.bmp | input image 2}"
                             "{@img3 | 18.bmp | input image 3}"
                             "{@nfeatures | 100000 | input number of features}"
                             "{@size | 3      | input threshold size}");
    
    // Imprimimos os parâmetros fornecidos
    cout << "img1 = " << parser.get<String>("@img1") 
         << "\nimg2 = " << parser.get<String>("@img2")
         << "\nimg3= " << parser.get<String>("@img3")
         << "\nnfeatures = " << parser.get<String>("@nfeatures") 
         << "\nsize = " << parser.get<String>("@size") << "\n";
    
    auto t2 = high_resolution_clock::now();

    // Calculando os milisegundos
    duration<double, std::milli> ms_double = t2 - t1;
	
	cout <<"Abrindo os arquivos= "
		 << ms_double.count() << "ms\n";
    
    t1 = high_resolution_clock::now();

	// Não usaremos as imagens de exemplo do opencv
    Mat img1 = imread( parser.get<String>("@img1"), IMREAD_GRAYSCALE);
    Mat img2 = imread( parser.get<String>("@img2"), IMREAD_GRAYSCALE);
    Mat img3 = imread( parser.get<String>("@img3"), IMREAD_GRAYSCALE);
    int nfeatures = parser.get<int>("@nfeatures");
    double tsize = parser.get<double>("@size");
    if ( img1.empty() || img2.empty() || img3.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        parser.printMessage();
        return -1;
    }
    
    t2 = high_resolution_clock::now();

    // Calculando o tempo em ms
    ms_double = t2 - t1;

    cout <<"Lendo os arquivos= "
		 << ms_double.count() << "ms\n";
    
    
    // O código segue o mesmo do tutorial com AKAZE
    vector<KeyPoint> kpts1, kpts2, kpts3;
    Mat desc1, desc2, desc3;
    Ptr<ORB> ORB = ORB::create(nfeatures);

    t1 = high_resolution_clock::now();
    ORB->detect(img1, kpts1, noArray());
    ORB->detect(img2, kpts2, noArray());
    ORB->detect(img3, kpts3, noArray());
    
    t2 = high_resolution_clock::now();
    
    //Escrevendo os pontos detectados
    
	ofstream kpoint1("kpts1_ORB.txt");
    
    for(size_t i = 0; i < kpts1.size(); i++) {
        kpoint1 << kpts1[i].pt.x << " " << kpts1[i].pt.y << endl;
    }
    
    kpoint1.close();
    
    ofstream kpoint2("kpts2_ORB.txt");
    
    for(size_t i = 0; i < kpts2.size(); i++) {
        kpoint2 << kpts2[i].pt.x << " " << kpts2[i].pt.y << endl;
    }
    
    kpoint2.close();

    ofstream kpoint3("kpts3_ORB.txt");
    
    for(size_t i = 0; i < kpts3.size(); i++) {
        kpoint3 << kpts3[i].pt.x << " " << kpts3[i].pt.y << endl;
    }
    
    kpoint3.close();
/* Getting number of milliseconds as a double. */
    ms_double = t2 - t1;

    cout <<"Detect= "
		 << ms_double.count() << "ms\n";
	    
    t1 = high_resolution_clock::now();
    
    ORB->compute(img1, kpts1, desc1);
    ORB->compute(img2, kpts2, desc2);
    ORB->compute(img3, kpts3, desc3);
    
    t2 = high_resolution_clock::now();
    
    /* Getting number of milliseconds as a double. */
    ms_double = t2 - t1;

    cout <<"Compute= "
		 << ms_double.count() << "ms\n";
    
    t1 = high_resolution_clock::now();
    
    //BFM 1 e 2
    
    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);
    vector<KeyPoint> matched1, matched2;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;
        if(dist1 < nn_match_ratio * dist2) {
            matched1.push_back(kpts1[first.queryIdx]);
            matched2.push_back(kpts2[first.trainIdx]);
        }
    }
    
    //BFM 2 e 3
		 
	//BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn2_matches;
    matcher.knnMatch(desc2, desc3, nn2_matches, 2);
    vector<KeyPoint> matched3, matched4;
    for(size_t i = 0; i < nn2_matches.size(); i++) {
        DMatch first = nn2_matches[i][0];
        float dist3 = nn2_matches[i][0].distance;
        float dist4 = nn2_matches[i][1].distance;
        if(dist3 < nn_match_ratio * dist4) {
            matched3.push_back(kpts2[first.queryIdx]);
            matched4.push_back(kpts3[first.trainIdx]);
        }
    }
    
    t2 = high_resolution_clock::now();
    
    // Calculando o tempo em ms
    ms_double = t2 - t1;

    cout <<"BFMatcher= "
		 << ms_double.count() << "ms\n";
    
    t1 = high_resolution_clock::now();
    
    // Computamos a matriz homográfica usando os pontos correlatos 
    vector<Point2f> mpts1, mpts2;
    KeyPoint::convert(matched1,mpts1);
    KeyPoint::convert(matched2,mpts2);
    Mat homography = findHomography(mpts1,mpts2,RANSAC, tsize);	
    
    
    vector<Point2f> mpts3, mpts4;
    KeyPoint::convert(matched3,mpts3);
    KeyPoint::convert(matched4,mpts4);
    Mat homography2 = findHomography(mpts3,mpts4,RANSAC, tsize);	
    
    t2 = high_resolution_clock::now();
 
    // Calculando o tempo em ms
    ms_double = t2 - t1;

    cout <<"Homography= "
		 << ms_double.count() << "ms\n";
    
    t1 = high_resolution_clock::now();
    
    // 1 e 2
    vector<DMatch> good_matches;
    vector<KeyPoint> inliers1, inliers2;
    for(size_t i = 0; i < matched1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_64F);
        col.at<double>(0) = matched1[i].pt.x;
        col.at<double>(1) = matched1[i].pt.y;
        col = homography * col;
        col /= col.at<double>(2);
        double dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
                            pow(col.at<double>(1) - matched2[i].pt.y, 2));
        if(dist < inlier_threshold) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }
    
    //2 e 3
    vector<DMatch> good_matches2;
    vector<KeyPoint> inliers3, inliers4;
    for(size_t i = 0; i < matched3.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_64F);
        col.at<double>(0) = matched3[i].pt.x;
        col.at<double>(1) = matched3[i].pt.y;
        col = homography2 * col;
        col /= col.at<double>(2);
        double dist = sqrt( pow(col.at<double>(0) - matched4[i].pt.x, 2) +
                            pow(col.at<double>(1) - matched4[i].pt.y, 2));
        if(dist < inlier_threshold) {
            int new_i = static_cast<int>(inliers3.size());
            inliers3.push_back(matched3[i]);
            inliers4.push_back(matched4[i]);
            good_matches2.push_back(DMatch(new_i, new_i, 0));
        }
    }
    
    t2 = high_resolution_clock::now();

    // Calculando o tempo em ms
    ms_double = t2 - t1;

    cout <<"Correlatos= "
		 << ms_double.count() << "ms\n";

    //Separando os pontos correlatos que não apareceriam nas 3 imagens e portanto não seriam inseridos no 3_images
    
    float coluna = img2.cols;
    
    vector<KeyPoint> smatches1,smatches2,smatches3,smatches4;
    for (size_t i=0; i < inliers2.size(); i++){
		if (inliers2[i].pt.x < coluna * 0.4){
			int new_i = static_cast<int>(smatches1.size());
            smatches1.push_back(inliers1[i]);
            smatches2.push_back(inliers2[i]);	
		}		
	}
	
	
	for (size_t i=0; i < inliers3.size(); i++){
		if (inliers3[i].pt.x > coluna * 0.6){
			int new_i = static_cast<int>(smatches3.size());
            smatches3.push_back(inliers3[i]);
            smatches4.push_back(inliers4[i]);	
		}		
	}

    
    //Comparando as listas de matches e criando a lista de 3 imagens
    
    vector<KeyPoint> pmatches1, pmatches2, pmatches3;    
    for (size_t i=0; i < inliers2.size(); i++){
        for(size_t c=0; c < inliers3.size(); c++){
            if (inliers2[i].pt.x == inliers3[c].pt.x && inliers2[i].pt.y == inliers3[c].pt.y) {
                int new_i = static_cast<int>(pmatches1.size());
                pmatches1.push_back(inliers1[i]);
                pmatches2.push_back(inliers2[i]);
                pmatches3.push_back(inliers4[c]);				
			}
		}		
	}

	size_t n = pmatches1.size()+smatches1.size()+smatches3.size();
    int k =15;
	ofstream bonsMatches("3_images_Matches_ORB.txt");
	ofstream ENH("3_images_ENH_ORB.txt");   
	
	for(size_t i = 0; i < smatches1.size(); i++) {
        bonsMatches << "1" << "\t" << k << "\t" << smatches1[i].pt.x << "\t" << smatches1[i].pt.y << "\n" << "2" << "\t" << k << "\t" << smatches2[i].pt.x << "\t" << smatches2[i].pt.y;
		ENH << k << "\t" << "Photogrammetric" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0";
		k++;
		if(i !=n-1){
			bonsMatches <<endl;
			ENH <<endl;
		}
    }
    
    for(size_t i = 0; i < smatches3.size(); i++) {
        bonsMatches << "2" << "\t" << k << "\t" << smatches3[i].pt.x << "\t" << smatches3[i].pt.y << "\n" << "3" << "\t" << k << "\t" << smatches4[i].pt.x << "\t" << smatches4[i].pt.y;
        ENH << k << "\t" << "Photogrammetric" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0";
		k++;
		if(i !=n-1){
			bonsMatches <<endl;
			ENH <<endl;
		}
    }
	
    for(size_t i = 0; i < pmatches1.size(); i++) {
        bonsMatches << "1" << "\t" << k << "\t" << pmatches1[i].pt.x << "\t" << pmatches1[i].pt.y << "\n" << "2" << "\t" << k << "\t" << pmatches2[i].pt.x << "\t" << pmatches2[i].pt.y << "\n" << "3" << "\t" << k << "\t" << pmatches3[i].pt.x << "\t" << pmatches3[i].pt.y;
		ENH << k << "\t" << "Photogrammetric" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0";
		k++;
		if(i !=n-1){
			bonsMatches <<endl;
			ENH <<endl;
		}			
    }
    
    bonsMatches.close();
    ENH.close();
	
    
    t1 = high_resolution_clock::now();
    
    Mat res;
    drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
    imwrite("ORB_result.png", res);
    
    t2 = high_resolution_clock::now();

    // Calculando o tempo em ms
    ms_double = t2 - t1;

    cout <<"Draw matches= "
		 << ms_double.count() << "ms\n";
    
    
    double inlier_ratio1 = inliers1.size() / (double) matched1.size();
    double inlier_ratio3 = inliers3.size() / (double) matched3.size();
    cout << "ORB Matching Results" << endl;
    cout << "*******************************" << endl;
    cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
    cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
    cout << "# Keypoints 3:                        \t" << kpts3.size() << endl;
    cout << "# Matches 1_2:                        \t" << matched1.size() << endl;
    cout << "# Matches 2_3:                        \t" << matched3.size() << endl;
    cout << "# Inliers 1_2:                        \t" << inliers1.size() << endl;
    cout << "# Inliers 2_3:                        \t" << inliers3.size() << endl;
    cout << "# Inliers Ratio 1_2:                  \t" << inlier_ratio1 << endl;
    cout << "# Inliers Ratio 2_3:                  \t" << inlier_ratio3 << endl;
    cout << "# 3 Matches:                          \t" << pmatches1.size() << endl;
    cout << endl;
    
    //imshow("result", res);
    waitKey();
    return 0;
}
