#include "NBC.h"
//#include <boost/algorithm/string.hpp>
struct dict {  
    string word;  
    double likelihood[NUM_CLASS];  
};  

int sortAccrodingto = 0;

int cmp(dict a, dict b)  {  
    return a.likelihood[sortAccrodingto] > b.likelihood[sortAccrodingto];  
}  

int main(int argc, char* argv[]) {

	NBC nbc;

	if (argc != 3){
		cout<< "Use format: ./NaiveBayes training_file test_file" << endl;
		exit(1);
	}

	string trainFile = "data/"+(string)argv[1];
	string testFile = "data/"+(string)argv[2];
	ofstream classout("word.txt");
	ifstream train(trainFile.c_str());
	ifstream test(testFile.c_str());

	

	for(int i = 0; i < NUM_CLASS; i++)
		nbc.numw_inclass[i] = 0;

	if(train.is_open() ){
		string line;
		int l =-1;
		vector<int> category;
		while (!train.eof()){
			getline(train, line, '\n');
			istringstream linestr(line);
		
			vector<string> words;
			string cate;
			linestr >> cate;
			
			string temp;
			while(linestr >> temp){
				int pos = temp.find(":");
				string word =  temp.substr(0, pos);
				int repeat_time = atoi(temp.substr(pos + 1, temp.length()).c_str());
			
			
				nbc.numw_inclass[atoi(cate.c_str())] += repeat_time;
			
				if(nbc.features.count(word) > 0){
					nbc.features[word].at(atoi(cate.c_str())) += repeat_time;
				}
				else{
					for(int k = 0; k < NUM_CLASS; k++)
						nbc.features[word].push_back(0);
					nbc.features[word].at(atoi(cate.c_str())) += repeat_time;
				}
			
				while(repeat_time--){
					words.push_back(word);
				}
			
			}

			nbc.ltrain.push_back(atoi(cate.c_str()));
			nbc.trainset.push_back(words);
		}
	}
	train.close();


	if(test.is_open()){
		string line;
		int l =-1;
		vector<int> category;
		while (!test.eof()){
			getline(test, line, '\n');
			istringstream linestr(line);
		
			vector<string> words;
			string cate;
			linestr >> cate;
			
			string temp;
			while(linestr >> temp){
				int pos = temp.find(":");
				string word =  temp.substr(0, pos);
				int repeat_time = atoi(temp.substr(pos + 1, temp.length()).c_str());
				while(repeat_time--){
					words.push_back(word);
				}
			}

			nbc.ltest.push_back(atoi(cate.c_str()));
			nbc.testset.push_back(words);
		}
	}
	test.close();

	// Start Training Naive Baysian Classifier
	nbc.Train(nbc.trainset, nbc.ltrain);
	nbc.Test(nbc.testset);

	list<dict> dictList;
	dict tmp;
	for (map<string,vector<double> >::iterator it = nbc.pTable.begin(); it!=nbc.pTable.end(); ++it){
        tmp.word = it->first;
		for(int i = 0; i < NUM_CLASS; i++){
			tmp.likelihood[i] = (it->second).at(i);  
		}
		dictList.push_back(tmp);
    }

	for(int j = 0; j < NUM_CLASS; j++){
		sortAccrodingto = j;
		cout<<"The top 20 words with the highest likelihood in class "<<j<<" are:"<<endl;
		dictList.sort(cmp); 
		int ct = 0;
		for (list<dict>::iterator it=dictList.begin(); it!=dictList.end(); ++it){
			cout<<it->word<<":"<<it->likelihood[sortAccrodingto]<<endl;
			if(ct < 19) {ct++; continue;}
			break;
		}
	}



	cout<<"The size of test data is "<<nbc.testset.size()<<endl;
}