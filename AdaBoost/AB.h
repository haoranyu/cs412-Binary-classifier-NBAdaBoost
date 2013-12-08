#ifndef __AB_H__
#define __AB_H__
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
using namespace std;

class AB {
	public:
		AB(string name);
		void train(const vector< vector<int> > & traindata, const vector<int> & train_label);
		void test(const vector< vector<int> > & testdata);
		void getTrainData(string path);
		void getTestData(string path);
		void printBasic(const vector<int> & label);
		void printDetail(const vector<int> & label);
		void printPtable();

		vector< vector<int> >	trainset, testset;
		vector<int> train_label, test_label;

	private:
		void calcuateMatrix(const vector<int> & label);
		int judge(const vector<int> &sample);
		int featureSize;
		int matrix[2][2];
		double prior[2];
		map<int, map<int, double> > pcp, ncp;
		vector< vector< vector<double> > > Ptable;
		vector<int> result, classCount, featureMax;
};

#endif