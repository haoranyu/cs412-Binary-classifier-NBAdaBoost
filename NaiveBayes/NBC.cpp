#include "NBC.h"
void NBC::Train( map< std::pair<int, int>, int > & fTable_p, map< std::pair<int, int>, int > & fTable_n, vector<int> & ltrain)
{
	int train_size = ltrain.size();
	int train_cate_size[NUM_CLASS];

	for(int i=0; i < NUM_CLASS; i++)
		train_cate_size[i] = 0;
	

	for (vector<int>::iterator it = ltrain.begin() ; it != ltrain.end(); ++it){
		train_cate_size[*it]++;
	}

	for(int i = 0; i < NUM_CLASS; i++){
		this->prior[i] = (double)train_cate_size[i]/(double)train_size;
	}

	for (std::map< std::pair<int, int>, int > ::iterator it = fTable_p.begin(); it != fTable_p.end(); ++it){
        vector<double> init;
        for (int i =0; i < NUM_CLASS; i++) {        
            init.push_back(0);
        }
        pTable.insert ( std::pair<std::pair<int, int>,vector<double> >(it->first, init) );
    }

	for (std::map<std::pair<int, int>, vector<double> >::iterator it = pTable.begin(); it != pTable.end(); ++it){
        (it->second).at(1) = (double)(fTable_p[it->first] + 1) / (double)(cate_size_train[1]);
        (it->second).at(0) = (double)(fTable_n[it->first] + 1) / (double)(cate_size_train[0]);
    }
}

void NBC::Test( vector< vector<int> > & data, vector<int> & ltest){
	int test_size = ltest.size();
	int confusion_matrix[NUM_CLASS][NUM_CLASS];

	for(int i = 0; i < NUM_CLASS; i++){
		for(int j = 0; j < NUM_CLASS; j++){
			confusion_matrix[i][j] = 0;
		}
	}

	for (int s = 0; s < test_size; s++) {
		int plabel = test(data[s]);
		this->pltest.push_back(plabel);
		confusion_matrix[ltest.at(s)][plabel]++;
	}
	
	int rowSum[NUM_CLASS];

	for(int i = 0; i < NUM_CLASS; i ++){
		rowSum[i] = 0;
	}

	for(int i = 0; i < NUM_CLASS; i ++){
		for(int j = 0; j < NUM_CLASS; j ++){
			rowSum[i] += confusion_matrix[i][j];
		}
	}

	cout<<"The confusion matrix:"<<endl;
	for(int i = 0; i < NUM_CLASS; i ++){
		for(int j = 0; j < NUM_CLASS; j ++){
			cout<<confusion_matrix[i][j]<<"\t";
		}
		cout<<endl;
	}

	cout<<confusion_matrix[1][1]<<" "<<confusion_matrix[1][0]<<" "<<confusion_matrix[0][1]<<" "<<confusion_matrix[0][0]<<endl;

	double 	accuracy = (double)(confusion_matrix[1][1] + confusion_matrix[0][0]) / (double)(confusion_matrix[1][1] + confusion_matrix[0][0] + confusion_matrix[1][0] + confusion_matrix[0][1]), 
	 		error = (double)(confusion_matrix[0][1] + confusion_matrix[1][0]) / (double)(confusion_matrix[1][1] + confusion_matrix[0][0] + confusion_matrix[1][0] + confusion_matrix[0][1]), 
			sensitivity = (double)(confusion_matrix[1][1]) / (double)(confusion_matrix[1][1] + confusion_matrix[1][0]), 
			specificity = (double)(confusion_matrix[0][0]) / (double)(confusion_matrix[0][0] + confusion_matrix[0][1]),
			precision = (double)(confusion_matrix[1][1]) / (double)(confusion_matrix[1][1] + confusion_matrix[0][1]);
	double 	&recall = sensitivity,
			f1 = 2 * (precision * recall) / (precision + recall),
			fhalf = (1 + 0.5*0.5) * (precision * recall) / (0.5*0.5 * precision + recall),
			f2 = (1 + 2*2) * (precision * recall) / (2*2 * precision + recall);

	// //cout<< accuracy <<" "<< error <<" "<< sensitivity <<" "<< specificity <<endl;
	// //cout<< precision <<" "<< f1 <<" "<< fhalf <<" "<< f2 <<endl;

	cout<<"Accuracy: "<<accuracy<<"\t\t";
	cout<<"Error Rate: "<<error<<endl;
	cout<<"Sensitivity: "<<sensitivity<<"\t";
	cout<<"Specificity: "<<specificity<<endl;
	cout<<"Precision: "<<precision<<"\t\t";
	cout<<"F-1 Score: "<<f1<<endl;
	cout<<"F-0.5 Score: "<<fhalf<<"\t";
	cout<<"F-2 Score: "<<f2<<endl;

}

int NBC::test(const vector<int> &tuple) {

    vector<double> posterior;
	double P[NUM_CLASS];
	for (int i = 0 ; i < NUM_CLASS; ++i){
		P[i] = log(this->prior[i]);
		//cout<<"P("<<this->prior[i]<<") and "<<P[i]<<endl;
	}
	for (int i = 0 ; i < NUM_CLASS; ++i){
		for(int f = 0; f < tuple.size(); ++f) {
			double pUnseen = (1.0f/(double)(cate_size_train[i] + cate_size_train[1] +  cate_size_train[0]));
			double pp;
			if(pTable.count(make_pair(f+1,tuple[f])) > 0){
				pp = this->pTable[std::make_pair(f+1,tuple[f])].at(i);
			}
			else{
				pp = pUnseen;
			}
			P[i] += log(pp);
		}
	}   

	return P[0] > P[1] ? 0:1; 

}
