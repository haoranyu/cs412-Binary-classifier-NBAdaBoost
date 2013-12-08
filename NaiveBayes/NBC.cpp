#include "NBC.h"

NBC::NBC(string name){
	if(name == "a1a.train"){
		this->featureSize = (123);
	}
	else if(name == "breast_cancer.train"){
		this->featureSize = (9);
	}
	else if(name == "led.train"){
		this->featureSize = (7);
	}
	else if(name == "poker.train"){
		this->featureSize = (10);
	}
}

void NBC::train(const vector< vector<int> > & traindata, const vector<int> & trainlabel) {
	this->counter[0] = this->counter[1] = 0;

	for (int j =0; j < this->featureSize; j++) {
		map<int, double> temp[2];			
		for (int i =0; i < traindata.size(); i++) {					
			if (trainlabel[i] == 1) {
				if (j == 0) counter[1]++;					
				if (temp[1].count( traindata[i][j] )==0)
					temp[1][traindata[i][j]] = double(1);
				else 
					temp[1][traindata[i][j]] += double(1); 

				if (temp[0].count( traindata[i][j] )==0)
					temp[0][traindata[i][j]] = double(0);
			}
			if (trainlabel[i] == -1) {
				if (j == 0) counter[0]++;					
				if (temp[0].count( traindata[i][j] )==0)
					temp[0][traindata[i][j]] = double(1);
				else 
					temp[0][traindata[i][j]] += double(1); 

				if (temp[1].count( traindata[i][j] )==0)
					temp[1][traindata[i][j]] = double(0);
			}		
		}
		bool null[2] = {false, false};
		double tp = 0;
		double tn = 0;
	   
		for (map<int,double>::iterator it = temp[1].begin(); it!=temp[1].end(); it++)	{
		    if (it->second == 0) {
		        null[1] = true;
		        break;
		    }
		}

		if (null[1] == true) {
		    for (map<int,double>::iterator it = temp[1].begin(); it!=temp[1].end(); it++) {
				temp[1][it->first] = double(temp[1][it->first]+1)/double(counter[1]+temp[1].size());
				tp += it->second;
		    }			
		}		

		if (null[1] == false) {
		    for (map<int,double>::iterator it = temp[1].begin(); it!=temp[1].end(); it++) {
		        temp[1][it->first] = double(temp[1][it->first])/double(counter[1]);
		        tp += it->second;
		    }
		}

		for (map<int,double>::iterator it = temp[0].begin(); it!=temp[0].end(); it++)	{
		    if (it->second == 0) {
			    null[0] = true;
			    break;
		    }
		}

		if (null[0] == true) {
		    for(map<int,double>::iterator it = temp[0].begin(); it!=temp[0].end(); it++) {
				temp[0][it->first] = double(temp[0][it->first]+1)/double(counter[0]+temp[0].size());
				tn += it->second;
			}
		}		

		if (null[0] == false)	{
		    for (map<int,double>::iterator it = temp[0].begin(); it!=temp[0].end(); it++) {
		        temp[0][it->first] = double(temp[0][it->first])/double(counter[0]);
		        tn += it->second;
		    }
		}
		
		this->pcp[j] = temp[1];
		this->ncp[j] = temp[0];		

	}
	this->prior[1] = double(counter[1])/double(counter[1]+counter[0]);
	this->prior[0] = double(counter[0])/double(counter[1]+counter[0]);
}

void NBC::test(const vector< vector<int> > & testdata ) {
	for (int s = 0; s < testdata.size(); s++) {
		this->result.push_back(judge(testdata[s]));
	}
}


int NBC::judge(const vector<int> &tuple) {
	double P = log(prior[1]);
	double N = log(prior[0]);

	for(int i = 0; i < featureSize; i++){
		P += log(pcp[i][tuple[i]]);
		N += log(ncp[i][tuple[i]]);
	}
	if (P > N) return 1;
	else return -1;
}

void NBC::getTrainData(string path){
	ifstream train(path.c_str());
	if(train.is_open()) {
		string line;			
		while (!train.eof()){
			getline(train, line, '\n');

			if(line == "") break;
			istringstream linestr(line);
		
			vector<int> tuple;
			int cate;
			linestr >> cate;


			this->trainlabel.push_back(cate);
			string temp;
			
			for(int i = 0; i < this->featureSize; i++){
				linestr >> temp;
				int pos = temp.find(":");
				int feature = atoi(temp.substr(0, pos).c_str());
				int value = atoi(temp.substr(pos + 1, temp.length()).c_str());

				while(feature != i && i < this->featureSize){
					tuple.push_back(0);
					i++;
				}

				if( i >= this->featureSize) break;

				tuple.push_back(value);
			}

			this->trainset.push_back(tuple);						
		}
	}
	train.close();
}
void NBC::getTestData(string path){
	ifstream test(path.c_str());
	if(test.is_open()) {
		string line;			
		while (!test.eof()){
			getline(test, line, '\n');

			if(line == "") break;
			istringstream linestr(line);
		
			vector<int> tuple;
			int cate;
			linestr >> cate;

			this->testlabel.push_back(cate);
			string temp;

			for(int i = 0; i < this->featureSize; i++){
				linestr >> temp;
				int pos = temp.find(":");
				int feature = atoi(temp.substr(0, pos).c_str());
				int value = atoi(temp.substr(pos + 1, temp.length()).c_str());

				while(feature != i && i < this->featureSize){
					tuple.push_back(0);
					i++;
				}

				if( i >= this->featureSize) break;

				tuple.push_back(value);
			}
			this->testset.push_back(tuple);						
		}
	}
	test.close();
}

void NBC::printBasic(const vector<int> & label){
	this->calcuateMatrix(label);
	cout<<matrix[1][1]<<" "<<matrix[1][0]<<" "<<matrix[0][1]<<" "<<matrix[0][0]<<endl;
}
void NBC::printDetail(const vector<int> & label){
	this->calcuateMatrix(label);
	double 	accuracy = (double)(matrix[1][1] + matrix[0][0]) / (double)(matrix[1][1] + matrix[0][0] + matrix[1][0] + matrix[0][1]), 
	 		error = (double)(matrix[0][1] + matrix[1][0]) / (double)(matrix[1][1] + matrix[0][0] + matrix[1][0] + matrix[0][1]), 
			sensitivity = (double)(matrix[1][1]) / (double)(matrix[1][1] + matrix[1][0]), 
			specificity = (double)(matrix[0][0]) / (double)(matrix[0][0] + matrix[0][1]),
			precision = (double)(matrix[1][1]) / (double)(matrix[1][1] + matrix[0][1]);
	double 	&recall = sensitivity,
			f1 = 2 * (precision * recall) / (precision + recall),
			fhalf = (1 + 0.5*0.5) * (precision * recall) / (0.5*0.5 * precision + recall),
			f2 = (1 + 2*2) * (precision * recall) / (2*2 * precision + recall);
	cout<<"Accuracy: "<<accuracy<<"\t\t";
	cout<<"Error Rate: "<<error<<endl;
	cout<<"Sensitivity: "<<sensitivity<<"\t";
	cout<<"Specificity: "<<specificity<<endl;
	cout<<"Precision: "<<precision<<"\t\t";
	cout<<"F-1 Score: "<<f1<<endl;
	cout<<"F-0.5 Score: "<<fhalf<<"\t";
	cout<<"F-2 Score: "<<f2<<endl;
}

void NBC::calcuateMatrix(const vector<int> & label){
	matrix[0][0] = matrix[0][1] = matrix[1][0] = matrix[1][1] =0;
	for (int n = 0; n < this->result.size(); n++) {
		if(label[n] == 1 && result[n] == 1) 
			matrix[1][1]++;
		if(label[n] == -1 && result[n] == -1)
			matrix[0][0]++;
		if(label[n] == -1 && result[n] == 1)
			matrix[0][1]++;
		if(label[n] == 1 && result[n] == -1)
			matrix[1][0]++;
	}
}