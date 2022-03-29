#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp> //Class for matching keypoint descriptors

using namespace std;
using namespace cv;

vector<Mat> Img_Open (int n) {
	
	String img;
	vector<Mat> images;

	for (size_t i = 0; i < n; i++){
		
		cout << "\n Insira o caminho para a Imagem: ";
		cin >> img;
		
		Mat Img = imread(img, IMREAD_GRAYSCALE);
		images.push_back(Img);
	    
	}
	return images;
}

vector<String> Img_Id (int n) {
	
	String Id;
	vector<String> Img_Id;

	for (size_t i = 0; i < n; i++){
		
		cout << "\n Insira o Id da Imagem: ";
		cin >> Id;
		
		Img_Id.push_back(Id);
	    
	}
	return Img_Id;
}
/*
int main(){
	
	int n;
	cout << "\n Insira a quantidade de Imagens: ";
	cin >> n;
	
	vector<Mat> read_images=Img_Open(n);
	
	imshow("Result", read_images[n-1]);
	waitKey();
	
}
*/