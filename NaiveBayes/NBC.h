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
#define NUM_CLASS 2

class NBC
{
	public:
		
		void Train( map< std::pair<int, int>, int > & fTable_p, map< std::pair<int, int>, int > & fTable_n, vector<int> & ltrain);
		void Test( vector< vector<int> > & testdata, vector<int> & ltest); 
		int test(const vector<int> &tuple);

		int featureSize;

		vector<int> ltrain, ltest, pltest;
		map< std::pair<int, int>, int > fTable_p_train, fTable_n_train, fTable_test;
		map< std::pair<int, int>, vector<double> > pTable;
		int cate_size_train[2];

		vector< vector<int> > 	trainset, testset ;
		double prior[NUM_CLASS];
		
	private:

};

#endif
