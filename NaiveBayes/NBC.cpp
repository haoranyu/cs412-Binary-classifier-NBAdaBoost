#include "NBC.h"

NBC::NBC(string name){
	ifstream filein(name.c_str());
	set<int> features;
	if(filein.is_open()) {
		string line;
		while (!filein.eof()){
			getline(filein, line, '\n');
			if(line == "") break;
			istringstream linestr(line);
			int cate;
			linestr >> cate;
			string temp;
			while(linestr >> temp){
				features.insert(atoi(temp.substr(0, temp.find(":")).c_str()));
			}
		}
	}
	filein.close();
	this->featureSize = (*features.rbegin());
}

void NBC::train(const vector< vector<int> > & traindata, const vector<int> & train_label) {
	for (int i =0; i < this->featureSize; i++) {
		featureMax.push_back(0);
	}

	for (int i = 0; i < this->featureSize; i++) {
		for (int j = 0; j < traindata.size(); j++) {
			 if(traindata[j][i] > featureMax[i]){
			 	featureMax[i] = traindata[j][i];
			}
		}
	}

	for (int j = 0; j < 2; j++) {
		vector< vector<double> > init;
		for (int i =0; i<this->featureSize; i++) {
			vector<double> inner;
			for (int k =0; k < featureMax[i]+1; k++) {
				inner.push_back(0.0);
			}	
			init.push_back(inner);
		}
		this->Ptable.push_back(init);
		this->classCount.push_back(0);
	}


	for (int j = 0; j < traindata.size(); j++) {
		for (int i = 0; i < this->featureSize; i++) {
			this->Ptable[train_label[j]][i][traindata[j][i]] += 1.0;
		}
		this->classCount[train_label[j]] ++;
	}
	
	for (int j = 0; j < 2; j++) {
		for (int i =0; i<this->featureSize; i++) {	
			for(int f = 0; f < featureMax[i]+1; f++){
				this->Ptable[j][i][f] = (this->Ptable[j][i][f])/(double)classCount[j];
			}
		}
		prior[j] = ((double)this->classCount[j]/(double)traindata.size());
	}
	printPtable();
}

void NBC::printPtable(){
	ofstream fout("featureMax.txt");
	fout<<"Table for -1"<<endl;
	for (int i =0; i<this->featureSize; i++) {
		for(int f = 0; f < featureMax[i]+1; f++){
			fout<<this->Ptable[0][i][f]<<"\t";
		}
		fout<<endl;
	}
	fout<<"Table for +1"<<endl;
	for (int i =0; i<this->featureSize; i++) {
		for(int f = 0; f < featureMax[i]+1; f++){
			fout<<this->Ptable[1][i][f]<<"\t";
		}
		fout<<endl;
	}
}

void NBC::test(const vector< vector<int> > & data) {
	this->result.clear();
	for (int s = 0; s<data.size(); s++) {		
		this->result.push_back(judge(data[s]));
	}	
}

void NBC::calcuateMatrix(const vector<int> & label){
	matrix[0][0] = matrix[0][1] = matrix[1][0] = matrix[1][1] =0;
	for (int n = 0; n < this->result.size(); n++) {
		this->matrix[label[n]][this->result[n]] +=1;
	}
}

int NBC::judge(const vector<int> &sample) {

    long double P[2];
    for (int i = 0 ; i< 2; ++i){
    	P[i] = log(this->prior[i]);
		for(int f = 0; f < featureSize; ++f) {
			long double pp;
			if(sample[f] <= featureMax[f]){
				pp = this->Ptable[i][f][sample[f]];
				if (pp < 0.0000001)
					pp = 0.0000001;
			}
			else{
				pp = 0.0000001;
			}
			P[i] += log(pp);
		}
	}
	return P[0] > P[1] ? 0 : 1; 
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


			this->train_label.push_back((cate < 0? 0:1));
			string temp;
			
			for(int i = 0; i < this->featureSize; i++){
				linestr >> temp;
				int pos = temp.find(":");
				int feature = atoi(temp.substr(0, pos).c_str());
				int value = atoi(temp.substr(pos + 1, temp.length()).c_str());

				while(feature-1 != i && i < this->featureSize){
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

			this->test_label.push_back((cate < 0? 0:1));
			string temp;

			for(int i = 0; i < this->featureSize; i++){
				linestr >> temp;
				int pos = temp.find(":");
				int feature = atoi(temp.substr(0, pos).c_str());
				int value = atoi(temp.substr(pos + 1, temp.length()).c_str());

				while(feature-1 != i && i < this->featureSize){
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