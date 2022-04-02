package core;

import core.network.Network;

public interface Fitness {

	double percentCorrect(Network net);
	double[][] getAnswers();
	double[][] getData();
}
