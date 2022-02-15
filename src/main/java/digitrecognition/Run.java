package digitrecognition;

import core.*;

public class Run {

    public static void main(String[] args) {

        long t0 = System.currentTimeMillis();

        // 98% achieved with:
        // 200 nodes in single hidden layer
        // 1500000 cycles
        // training rate of 300
        // stochastic batch size 1

        // can also get 98% in about a half hour on my machine with:
        // 200 nodes in single hidden layer
        // 150000 cycles
        // training rate 400
        // stochastic batch size 10

        int cycles = 500000;
        double trainingRate = 700; // how well this works depends on values below...!
        boolean printWrong = true; // on test, print those it got wrong

        double percentToDo = 0.01; // use 0.01 to test regular GD if not stochastic, this is handled below automatically
        boolean onOffPixels = false; // probably should be false
        boolean stochastic = true; // false for testing and percentToDo = 0.001, displays running mse but VERY slow
        int stochasticBatchSize = 1; // can divide cycles by this number and gain some speed

        int ram = 10; // how many previous values to keep track of Running Avg Mse, 0 = don't do this at all

        if (stochastic) {
            percentToDo = 1.0;
        } else {
            percentToDo = 0.0005;
        }
//        Network net = new BasicNetwork(new BasicNetworkParameters(28 * 28, 10));
        Network net = new LayeredNetwork(new LayeredNetworkParameters(new int[]{200, 40}, 28 * 28, 10));
        DigitRecognitionFitness trainFit = new DigitRecognitionFitness(true, percentToDo, onOffPixels, false);
        DigitRecognitionFitness testFit = new DigitRecognitionFitness(false, 1.0, onOffPixels, printWrong);

        Trainer t = new Trainer(trainingRate, net, trainFit, percentToDo, stochastic, stochasticBatchSize, ram);

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
