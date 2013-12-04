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
#include <list>
using namespace std;
#define NUM_CLASS 2

class AB
{
	public:
		
		void Train(vector< vector<string> > & traindata,  vector<int> & ltrain);

		void Sample();
		void normalize(); 

		void Test( vector< vector<string> > & testdata, vector<int> & ltest); 
		int test(const vector<string> &sample);


		vector< vector<string> >	trainset, testset ;

		vector<int> ltrain, ltest, pltest;

		vector< vector<string> >	sampleset;
		vector<int> lsample;
		vector<int> sample_map;
		vector<double> weight;


		map<string, vector<double> > pTable;
		map<string, vector<int> > features;
		
		double prior[NUM_CLASS];
		long 	numw_inclass[NUM_CLASS];
	
	private:

};

#endif
