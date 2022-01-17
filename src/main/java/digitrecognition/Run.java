package digitrecognition;

import core.*;

public class Run {

    public static void main(String[] args) {

        long t0 = System.currentTimeMillis();

        int cycles = 100000;
        double trainingRate = 300;

        double percentToDo = 1.0; // do 1.0 for stochastic, otherwise 0.01 to test regular GD
        boolean onOffPixels = false;
        boolean stochastic = true; // making this false seems to require/allow higher trainingRate? unsure

//        Network net = new BasicNetwork(new BasicNetworkParameters(28 * 28, 10));
        Network net = new LayeredNetwork(new LayeredNetworkParameters(new int[]{200}, 28 * 28, 10));
        DigitRecognitionFitness testFit = new DigitRecognitionFitness(false, 1.0, onOffPixels);
        DigitRecognitionFitness trainFit = new DigitRecognitionFitness(true, percentToDo, onOffPixels);

        Trainer t = new Trainer(trainingRate, net, trainFit, percentToDo, stochastic);

        double[][] data = t.train(cycles);

        if (!stochastic) {
            DataLineChart chart = new DataLineChart(
                    "Digit Recognizer Training Data",
                    "",
                    data,
                    cycles,
                    0.0,
                    1.0,
                    "Average MSE",
                    "Percent Correct");
            chart.pack();
            chart.setVisible(true);
        }

        long t1 = System.currentTimeMillis();

        double finalTestScore = testFit.percentCorrect(net);
        double finalTrainingScore = trainFit.percentCorrect(net);

        System.out.println("Final percent correct on training data = " + 100.0 * finalTrainingScore);
        System.out.println("Final percent correct on test data = " + 100.0 * finalTestScore);

        System.out.println("Total time = " + (t1-t0) + " ms");
    }
}
