#include "NBC.h"

int main(int argc, char* argv[]) {

	if (argc != 3){
			cout<< "Use format: ./NaiveBayes training_file test_file" << endl;
			exit(1);
	}

	string train = "data/"+(string)argv[1];
	string test = "data/"+(string)argv[2];

	NBC nbc((string)argv[1]);

	nbc.getTrainData(train);
	nbc.getTestData(test);

	nbc.train(nbc.trainset, nbc.train_label);
	nbc.test(nbc.testset);
	nbc.printBasic(nbc.test_label);
	nbc.printDetail(nbc.test_label);

	NBC nbc2((string)argv[1]);
	nbc2.getTrainData(train);
	nbc2.getTestData(train);

	nbc2.train(nbc2.trainset, nbc2.train_label);
	nbc2.test(nbc2.testset);
	nbc2.printBasic(nbc2.test_label);
	nbc2.printDetail(nbc2.test_label);

}