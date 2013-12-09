#include "AB.h"
#define K 7
AB::AB(string name){
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

void AB::rewieght(int size){
	double sumWeight = 0;
	for(int i = 0; i < size; i++){
		sumWeight += this->weight[i];
	}
	for(int i = 0; i < size; i++){
		this->weight[i] = this->weight[i] / (double)sumWeight;
	}
}

void AB::sampling(const vector< vector<int> > & data, vector<double> & weight , vector<int>& label) {
    vector< vector<int> >  temp;
	vector<int> ltmp;

    label.clear();
    double n =0;
	double rn =0;
    for (int i =0; i<data.size(); i++) {
		rn = rand() / double(RAND_MAX);
		n = 0;
		for (int j =0; j<data.size(); j++) {
			n+= weight[j];
			if (n > rn)	{
				temp.push_back(data[j]);
				ltmp.push_back(label[j]);
				break;
			}
		}
    } 
    this->sampleset = temp;
    this->sample_label = ltmp;
}

void AB::updateWeight(double error, int k){
	vector<double>	temp_weight;
	for (int i = 0; i < this->sample_label.size(); i++) {
	 	if(this->sample_label[i] != this->result[k][i]){
	 		temp_weight.push_back( this->weight[i] * error / (1.0f-error));
	 	}
	 	else{
	 		temp_weight.push_back( this->weight[i]);
	 	}
	}
	this->weight.clear();
	this->weight = temp_weight;
}

void AB::adaBoostTrain(const vector< vector<int> > & data, vector<int> & label){
	for(int i = 0; i < label.size(); i++){
		this->weight.push_back(1);
	}
	rewieght(label.size());

	for(int k = 0; k < K; k++){
		vector<int> init;
		for (int i = 0; i < this->sample_label.size(); i++)
			init.push_back(0);
		this->result.push_back(init);
	}

	for(int k = 0; k < K; k++){
		sampling(data, this->weight ,label);
		train(this->sampleset, this->sample_label, k);
		test(this->sampleset, k);
		computeError(k);
		if(this->error[k] > 0.5){
			this->Ptable.pop_back();
			this->error.pop_back();
			this->featureMax.pop_back();
			k--;
			continue;
		}
		updateWeight(this->error[k] ,k);
		rewieght(label.size());
	}
}

int AB::adaBoostJudge(const vector<int> &sample){
	double classWeight[2] = {0.0f,0.0f};
	for(int k = 0; k < K; k++){
		double w_i = ((1.0f - this->error[k])/this->error[k]);
		int pred = judge(sample, k);
		classWeight[pred] += w_i;
	}
	return classWeight[0] > classWeight[1]? 0 : 1;
}

void AB::adaBoostTest(const vector< vector<int> > & data) {
	this->adaResult.clear();
	for (int s = 0; s<data.size(); s++) {		
		this->adaResult.push_back(adaBoostJudge(data[s]));
	}
}

void AB::computeError(int k){
	double temp_error = 0.0f;
	for(int i = 0; i < this->sampleset.size(); i++){
		if(this->sample_label[i] != this->result[k][i]){
			temp_error += this->weight[i];
		}
	}
	this->error.push_back(temp_error);
}


void AB::train(const vector< vector<int> > & traindata, const vector<int> & train_label, int k) {
	this->classCount.clear();
	vector<int> init;
	vector<vector<vector<double> > > Ptable_k;
	for (int i =0; i < this->featureSize; i++) {
		init.push_back(0);
	}
	featureMax.push_back(init);

	for (int i = 0; i < this->featureSize; i++) {
		for (int j = 0; j < traindata.size(); j++) {
			 if(traindata[j][i] > featureMax[k][i]){
			 	featureMax[k][i] = traindata[j][i];
			}
		}
	}

	for (int j = 0; j < 2; j++) {
		vector< vector<double> > init;
		for (int i =0; i<this->featureSize; i++) {
			vector<double> inner;
			for (int h =0; h < featureMax[k][i]+1; h++) {
				inner.push_back(0.0);
			}	
			init.push_back(inner);
		}
		
		Ptable_k.push_back(init);

		this->classCount.push_back(0);
	}


	for (int j = 0; j < traindata.size(); j++) {
		for (int i = 0; i < this->featureSize; i++) {
			Ptable_k[train_label[j]][i][traindata[j][i]] += 1.0;
		}
		this->classCount[train_label[j]] ++;
	}
	
	for (int j = 0; j < 2; j++) {
		for (int i =0; i<this->featureSize; i++) {	
			for(int f = 0; f < featureMax[k][i]+1; f++){
				Ptable_k[j][i][f] = Ptable_k[j][i][f]/(double)classCount[j];
			}
		}
		prior[k][j] = ((double)this->classCount[j]/(double)traindata.size());
	}
	this->Ptable.push_back(Ptable_k);
}

void AB::test(const vector< vector<int> > & data, int k) {
	this->result[k].clear();
	for (int s = 0; s<data.size(); s++) {		
		this->result[k].push_back(judge(data[s], k));
	}
}

void AB::calcuateMatrix(const vector<int> & label){
	matrix[0][0] = matrix[0][1] = matrix[1][0] = matrix[1][1] =0;
	for (int n = 0; n < this->adaResult.size(); n++) {
		this->matrix[label[n]][this->adaResult[n]] +=1;
	}
}

int AB::judge(const vector<int> &sample, int k) {

    long double P[2];
    for (int i = 0 ; i< 2; ++i){
    	P[i] = log(this->prior[k][i]);
		for(int f = 0; f < featureSize; ++f) {
			long double pp;
			if(sample[f] <= featureMax[k][f]){
				pp = this->Ptable[k][i][f][sample[f]];
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


void AB::getTrainData(string path){
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
void AB::getTestData(string path){
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

void AB::printBasic(const vector<int> & label){
	this->calcuateMatrix(label);
	cout<<matrix[1][1]<<" "<<matrix[1][0]<<" "<<matrix[0][1]<<" "<<matrix[0][0]<<endl;
}
void AB::printDetail(const vector<int> & label){
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