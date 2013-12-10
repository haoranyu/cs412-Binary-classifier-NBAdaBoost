#include "AB.h"

int main(int argc, char* argv[]) {

	if (argc != 3){
			cout<< "Use format: ./NaiveBayes training_file test_file" << endl;
			exit(1);
	}

	string train = (string)argv[1];
	string test = (string)argv[2];

	AB ab(train);

	ab.getTrainData(train);
	ab.getTestData(test);

	ab.adaBoostTrain(ab.trainset, ab.train_label);

	ab.adaBoostTest(ab.trainset);
	ab.printBasic(ab.train_label);
	//ab.printDetail(ab.train_label);

	ab.adaBoostTest(ab.testset);
	ab.printBasic(ab.test_label);
	//ab.printDetail(ab.test_label);

}