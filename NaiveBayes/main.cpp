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

	nbc.test(nbc.trainset);
	nbc.printBasic(nbc.train_label);
	nbc.printDetail(nbc.train_label);

	nbc.test(nbc.testset);
	nbc.printBasic(nbc.test_label);
	nbc.printDetail(nbc.test_label);

}