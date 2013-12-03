#include "NBC.h"
string mapIdx(string str){
	if(str == "-1"){
		return "0";
	}
	else if(str == "+1"){
		return "1";
	}
	else{
		return str;
	}
}

int main(int argc, char* argv[]) {

	NBC nbc;

	if (argc != 3){
			cout<< "Use format: ./NaiveBayes training_file test_file" << endl;
			exit(1);
	}

	string trainFile = "data/"+(string)argv[1];
	string testFile = "data/"+(string)argv[2];

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
			cate = mapIdx(cate);
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
			cate = mapIdx(cate);
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
}