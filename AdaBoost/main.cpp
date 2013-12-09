#include "AB.h"

int main(int argc, char* argv[]) {

	if (argc != 3){
			cout<< "Use format: ./NaiveBayes training_file test_file" << endl;
			exit(1);
	}

	string train = "data/"+(string)argv[1];
	string test = "data/"+(string)argv[2];

	AB ab((string)argv[1]);

	ab.getTrainData(train);
	ab.getTestData(test);

	ab.adaBoostTrain(ab.trainset, ab.train_label);
	ab.adaBoostTest(ab.trainset);
	//ab.test(ab.testset);
	ab.printBasic(ab.train_label);
	ab.printDetail(ab.train_label);

	// ab.test(ab.trainset);
	// ab.printBasic(ab.train_label);
	// ab.printDetail(ab.train_label);


}