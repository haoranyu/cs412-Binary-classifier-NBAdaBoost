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
#define K 7
using namespace std;

class AB {
	public:
		AB(string name);
		void 	adaBoostTrain(const vector< vector<int> > & data, vector<int> & label);
		void	adaBoostTest(const vector< vector<int> > & data);
		int 	adaBoostJudge(const vector<int> &sample);
		void 	train(const vector< vector<int> > & traindata, const vector<int> & train_label, int k);
		void 	test(const vector< vector<int> > & testdata, int k);
		void 	getTrainData(string path);
		void 	getTestData(string path);
		void 	printBasic(const vector<int> & label);
		void 	printDetail(const vector<int> & label);

		vector< vector<int> >	trainset, 
								testset,
								sampleset;
		vector<int> 			train_label, 
								test_label,
								sample_label;

	private:
		void	sampling(const vector< vector<int> > & data, vector<double> & weight , vector<int>& label);
		void	updateWeight(double error, int k);
		void 	computeError(int k);
		void 	calcuateMatrix(const vector<int> & label);
		int 	judge(const vector<int> &sample, int k);
		void 	rewieght(int size);

		int 								featureSize;
		int 								matrix[2][2];
		double 								prior[K][2];
		map<int, map<int, double> > 		pcp, 
											ncp;
		vector<vector<vector<vector<double> > > >	Ptable;
		vector<vector<int> >				result,
											featureMax;
		vector<int>							classCount,
											adaResult; 
		vector<double>						weight,
											error;
};

#endif