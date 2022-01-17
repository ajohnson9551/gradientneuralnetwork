package core;

public interface Fitness {

    double mse(Network net);
    double percentCorrect(Network net);
    double[][] getAnswers();
    double[][] getData();
}
