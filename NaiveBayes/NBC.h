#ifndef __NBC_H__
#define __NBC_H__
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>	
#include <map>
#include <list>
using namespace std;

class NBC {
	public:
		NBC(string name);
		void train(const vector< vector<int> > & traindata, const vector<int> & trainlabel);
		void test(const vector< vector<int> > & testdata);
		void getTrainData(string path);
		void getTestData(string path);
		void printBasic(const vector<int> & label);
		void printDetail(const vector<int> & label);
		
		vector< vector<int> >	trainset, testset;
		vector<int> trainlabel, testlabel, result;
		
	private:
		int judge(const vector<int> &sample);
		void calcuateMatrix(const vector<int> & label);
		int featureSize;
		int matrix[2][2];
		map<int, map<int, double> > pcp, ncp;
		double prior[2];
		int counter[2];
};

#endif
