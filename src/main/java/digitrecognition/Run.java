package digitrecognition;

import core.*;

public class Run {

    public static void main(String[] args) {

        long t0 = System.currentTimeMillis();

        // BELOW IS WHEN YOU ARE USING trainStep1 METHOD IN THE TRAINER CLASS, NOW DEPRECATED AND UNUSED

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

        // BELOW IS USING trainStep2 METHOD WITH TWO LAYERS, THIS IS THE CURRENT IMPLEMENTATION

        // 120000 cycles
        // 500 training rate
        // 10 batch size
        // 0 momentum
        // 200 40 layers
        // expAcc false
        // 6000 seconds, 98.1% test (99.6% training, maybe a bit of overtraining)

        // 360000 cycles
        // 500 training rate
        // 20 batch size
        // 0 momenutum
        // 250 50 layers
        // expAcc false
        // 24522 seconds, 98.4% test (99.9% training)


        int cycles = 120000;
        double trainingRate = 400; // how well this works depends on values below...!
        boolean printWrong = true; // on test, print those it got wrong

        double percentToDo = 0.01; // use 0.01 to test regular GD if not stochastic, this is handled below automatically
        boolean onOffPixels = false; // probably should be false
        boolean stochastic = true; // false for testing and percentToDo = 0.001, displays running mse but VERY slow
        int stochasticBatchSize = 10; // can divide cycles by this number and gain some speed

        double momentum = 0; // constant for momentum gradient descent, set to 0 to disable, definitely < 1 and > 0... maybe should be 0, causing divergences...

        int ram = 10; // how many previous values to keep track of Running Avg Mse, 0 = don't do this at all
        boolean expAcc = false; // use experimental acceleration, should lower trainingRate and must have ram > 0 (doesn't work well...)

        if (expAcc && ram == 0) {
            ram = 10;
        }

        if (stochastic) {
            percentToDo = 1.0;
        } else {
            percentToDo = 0.0005;
        }
//        Network net = new BasicNetwork(new BasicNetworkParameters(28 * 28, 10));
        Network net = new LayeredNetwork(new LayeredNetworkParameters(new int[]{200, 40}, 28 * 28, 10));
        DigitRecognitionFitness trainFit = new DigitRecognitionFitness(true, percentToDo, onOffPixels, false);
        DigitRecognitionFitness testFit = new DigitRecognitionFitness(false, 1.0, onOffPixels, printWrong);

        Trainer t = new Trainer(trainingRate, net, trainFit, percentToDo, stochastic, stochasticBatchSize, ram, expAcc, momentum);

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
