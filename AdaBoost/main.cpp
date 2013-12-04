#include "AB.h"
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

	AB ab;

	if (argc != 3){
			cout<< "Use format: ./NaiveBayes training_file test_file" << endl;
			exit(1);
	}

	string trainFile = "data/"+(string)argv[1];
	string testFile = "data/"+(string)argv[2];

	ifstream train(trainFile.c_str());
	ifstream test(testFile.c_str());

	for(int i = 0; i < NUM_CLASS; i++)
		ab.numw_inclass[i] = 0;

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
			
			
				ab.numw_inclass[atoi(cate.c_str())] += repeat_time;
			
				if(ab.features.count(word) > 0){
					ab.features[word].at(atoi(cate.c_str())) += repeat_time;
				}
				else{
					for(int k = 0; k < NUM_CLASS; k++)
						ab.features[word].push_back(0);
					ab.features[word].at(atoi(cate.c_str())) += repeat_time;
				}
			
				while(repeat_time--){
					words.push_back(word);
				}
			
			}

			ab.ltrain.push_back(atoi(cate.c_str()));
			ab.trainset.push_back(words);
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

			ab.ltest.push_back(atoi(cate.c_str()));
			ab.testset.push_back(words);
		}
	}
	test.close();

	// Start Training Naive Baysian Classifier
	ab.Train(ab.trainset, ab.ltrain);
	ab.Test(ab.trainset, ab.ltrain);
	ab.Test(ab.testset, ab.ltest);
}