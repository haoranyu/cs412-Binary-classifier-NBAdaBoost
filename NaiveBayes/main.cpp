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

	if((string)argv[1] == "a1a.train"){
		nbc.featureSize = 123;
	}
	else if((string)argv[1] == "breast_cancer.train"){
		nbc.featureSize = 9;
	}
	else if((string)argv[1] == "led.train"){
		nbc.featureSize = 7;
	}
	else if((string)argv[1] == "poker.train"){
		nbc.featureSize = 10;
	}

	ifstream train(trainFile.c_str());
	ifstream test(testFile.c_str());

	if(train.is_open() ){
		string line;
		int l =-1;
		vector<int> category;
		nbc.cate_size_train[0] = 0;
		nbc.cate_size_train[1] = 0;

		while (!train.eof()){
			getline(train, line, '\n');

			if(line == "") break;
			istringstream linestr(line);
		
			vector<int> tuple;
			string cate;
			linestr >> cate;
			cate = mapIdx(cate);
			string temp;
			for(int i = 0; i < nbc.featureSize; i++)
            	tuple.push_back(0);
            int ct = 1;
			while(linestr >> temp){

				int pos = temp.find(":");
				int feature = atoi(temp.substr(0, pos).c_str());
				int value = atoi(temp.substr(pos + 1, temp.length()).c_str());



				while(ct != feature){
					if(atoi(cate.c_str()) == 1){
						if(nbc.fTable_p_train.count(make_pair(ct, 0)) == 0){
							nbc.fTable_p_train.insert( std::pair<std::pair<int,int>, int>(make_pair(ct, 0), 1));
							nbc.fTable_n_train.insert( std::pair<std::pair<int,int>, int>(make_pair(ct, 0), 0));
						}
						else{
							nbc.fTable_p_train[make_pair(ct, 0)]++;
						}
					}
					else{
						if(nbc.fTable_n_train.count(make_pair(ct, 0)) == 0){
							nbc.fTable_n_train.insert( std::pair<std::pair<int,int>, int>(make_pair(ct, 0), 1));
							nbc.fTable_p_train.insert( std::pair<std::pair<int,int>, int>(make_pair(ct, 0), 0));
						}
						else{
							nbc.fTable_n_train[make_pair(ct, 0)]++;
						}
					}
					ct++;
				}
				
				tuple.at(feature - 1) = value;

				if(atoi(cate.c_str()) == 1){
					if(nbc.fTable_p_train.count(make_pair(feature, value)) == 0){
						nbc.fTable_p_train.insert( std::pair<std::pair<int,int>, int>(make_pair(feature, value), 1));
						nbc.fTable_n_train.insert( std::pair<std::pair<int,int>, int>(make_pair(feature, value), 0));
					}
					else{
						nbc.fTable_p_train[make_pair(feature, value)]++;
					}
				}
				else{
					if(nbc.fTable_n_train.count(make_pair(feature, value)) == 0){
						nbc.fTable_n_train.insert( std::pair<std::pair<int,int>, int>(make_pair(feature, value), 1));
						nbc.fTable_p_train.insert( std::pair<std::pair<int,int>, int>(make_pair(feature, value), 0));
					}
					else{
						nbc.fTable_n_train[make_pair(feature, value)]++;
					}
				}
				ct++;
			}

			nbc.cate_size_train[atoi(cate.c_str())] ++; 
			nbc.ltrain.push_back(atoi(cate.c_str()));
			nbc.trainset.push_back(tuple);
		}
	}
	train.close();


	if(test.is_open()){
		string line;
		int l =-1;

		while (!test.eof()){
            getline(test, line, '\n');
            istringstream linestr(line);
    
            vector<int> tuple;
            string cate;
            linestr >> cate;
            cate = mapIdx(cate);
            string temp;

            //cout<<nbc.featureSize<<endl;
            for(int i = 0; i < nbc.featureSize; i++)
            	tuple.push_back(0);

            while(linestr >> temp){
                int pos = temp.find(":");
                int feature = atoi(temp.substr(0, pos).c_str());
				int value = atoi(temp.substr(pos + 1, temp.length()).c_str());
                tuple.at(feature - 1) = value;
            }

            nbc.ltest.push_back(atoi(cate.c_str()));
            nbc.testset.push_back(tuple);
        }
	}
	test.close();

	// Start Training Naive Baysian Classifier
	nbc.Train(nbc.fTable_p_train, nbc.fTable_n_train, nbc.ltrain);
	nbc.Test(nbc.trainset, nbc.ltrain);
	nbc.Test(nbc.testset, nbc.ltest);

}