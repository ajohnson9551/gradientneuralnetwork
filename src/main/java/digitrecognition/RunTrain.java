package digitrecognition;

import core.*;
import core.layer.*;
import core.network.ConvolutionalNetwork;
import core.network.ConvolutionalNetworkParameters;
import core.network.Network;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class RunTrain {

	public static void main(String[] args) {

		long t0 = System.currentTimeMillis();

		int cycles = 12000;
		double trainingRate = 30;
		int stochasticBatchSize = 2;
		int ram = 10;

		List<LayerParameters> layerParams = new ArrayList<>(List.of(
				new ConvolutionalLayerParameters(3, 4, -1, ActFunc.RELU),
				new PoolLayerParameters(2, 2),
				new FullLayerParameters(100, ActFunc.SIGMOID),
				new FullLayerParameters(10, ActFunc.SIGMOID)
		));

		ConvolutionalNetworkParameters netParams = new ConvolutionalNetworkParameters(new int[]{28, 28, 1}, 10, layerParams, stochasticBatchSize);
		Network net = new ConvolutionalNetwork(netParams);

		DigitRecognitionFitness trainFit = new DigitRecognitionFitness(true, 1.0, false);
		DigitRecognitionFitness testFit = new DigitRecognitionFitness(false, 1.0, true);

		Trainer t = new Trainer(trainingRate, net, trainFit, stochasticBatchSize, ram);

		t.train(cycles);

		long t1 = System.currentTimeMillis();

		double finalTestScore = testFit.percentCorrect(net);
		double finalTrainingScore = trainFit.percentCorrect(net);

		System.out.println("Final percent correct on training data = " + 100.0 * finalTrainingScore);
		System.out.println("Final percent correct on test data = " + 100.0 * finalTestScore);

		System.out.println("Total time = " + (t1 - t0) + " ms");

		System.out.println("Save network? y/n");
		if (askUserToContinue()) {
			String path = "networks";
			net.serialize(path);
		}
	}

	public static boolean askUserToContinue() {
		Scanner scan = new Scanner(System.in);
		String input = scan.nextLine();
		return input.equalsIgnoreCase("y");
	}
}
