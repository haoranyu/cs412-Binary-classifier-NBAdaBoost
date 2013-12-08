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

	ab.train(ab.trainset, ab.train_label);
	ab.test(ab.testset);
	ab.printBasic(ab.test_label);
	ab.printDetail(ab.test_label);

	AB ab2((string)argv[1]);
	ab2.getTrainData(train);
	ab2.getTestData(train);

	ab2.train(ab2.trainset, ab2.train_label);
	ab2.test(ab2.testset);
	ab2.printBasic(ab2.test_label);
	ab2.printDetail(ab2.test_label);

}