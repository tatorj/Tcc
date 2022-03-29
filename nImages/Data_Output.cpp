#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp> //Class for matching keypoint descriptors
#include "fstream"

using namespace std;
using namespace cv;

void Data_Output_Akaze (int n, int P, vector<vector<vector<KeyPoint>>> pmatches, vector<vector<vector<KeyPoint>>> smatches /*,vector<String> Img_Id*/){

	P++;
	size_t f = pmatches.size()+smatches.size();
	
	ofstream bonsMatches("3_images_Matches_AKAZE.txt");
	ofstream ENH("3_images_ENH_AKAZE.txt");

	for (int k = 0; k < n-1; ++k)
	{
		for(size_t i = 0; i < smatches[k].size(); i++) 
		{
			bonsMatches << "1" << "\t" << P << "\t" << smatches[k][0][i].pt.x << "\t" << smatches[k][0][i].pt.y << "\n" << "2" << "\t" << P << "\t" << smatches[k][1][i].pt.x << "\t" << smatches[k][1][i].pt.y;
			ENH << P << "\t" << "Photogrammetric" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0";
			P++;

			if(i !=f-1)
			{
				bonsMatches <<endl;
				ENH <<endl;
			}	
		}
		for(size_t i = 0; i < pmatches[k].size(); i++) 
		{
			bonsMatches << "1" << "\t" << P << "\t" << pmatches[k][0][i].pt.x << "\t" << pmatches[k][0][i].pt.y << "\n" << "2" << "\t" << P << "\t" << pmatches[k][1][i].pt.x << "\t" << pmatches[k][1][i].pt.y << "\n" << "3" << "\t" << P << "\t" << pmatches[k][2][i].pt.x << "\t" << pmatches[k][2][i].pt.y;
			ENH << P << "\t" << "Photogrammetric" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0";
			P++;

			if(i !=f-1)
			{
				bonsMatches <<endl;
				ENH <<endl;
			}			
		}
	}

	bonsMatches.close();
	ENH.close();
	return;

}



/*
	size_t n = pmatches1.size()+smatches1.size()+smatches3.size();
	int P =15;
	ofstream bonsMatches("3_images_Matches_SIFT.txt");
	ofstream ENH("3_images_ENH_SIFT.txt");   
	
	for(size_t i = 0; i < smatches1.size(); i++) {
		bonsMatches << "1" << "\t" << P << "\t" << smatches1[i].pt.x << "\t" << smatches1[i].pt.y << "\n" << "2" << "\t" << P << "\t" << smatches2[i].pt.x << "\t" << smatches2[i].pt.y;
		ENH << P << "\t" << "Photogrammetric" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0";
		P++;
		if(i !=n-1){
			bonsMatches <<endl;
			ENH <<endl;
		}
	}
	
	for(size_t i = 0; i < smatches3.size(); i++) {
		bonsMatches << "2" << "\t" << P << "\t" << smatches3[i].pt.x << "\t" << smatches3[i].pt.y << "\n" << "3" << "\t" << P << "\t" << smatches4[i].pt.x << "\t" << smatches4[i].pt.y;
		ENH << P << "\t" << "Photogrammetric" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0";
		P++;
		if(i !=n-1){
			bonsMatches <<endl;
			ENH <<endl;
		}
	}
	
	for(size_t i = 0; i < pmatches1.size(); i++) {
		bonsMatches << "1" << "\t" << P << "\t" << pmatches1[i].pt.x << "\t" << pmatches1[i].pt.y << "\n" << "2" << "\t" << P << "\t" << pmatches2[i].pt.x << "\t" << pmatches2[i].pt.y << "\n" << "3" << "\t" << P << "\t" << pmatches3[i].pt.x << "\t" << pmatches3[i].pt.y;
		ENH << P << "\t" << "Photogrammetric" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0" << "\t" << "0";
		P++;
		if(i !=n-1){
			bonsMatches <<endl;
			ENH <<endl;
		}			
	}
	
	bonsMatches.close();
	ENH.close();
	*/