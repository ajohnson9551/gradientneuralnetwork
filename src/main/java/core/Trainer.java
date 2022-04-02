package core;

import core.network.ConvolutionalNetwork;
import core.network.Network;

import java.util.Random;

public class Trainer {

	private final double trainingRate;
	private final Network net;

	private final int ram;
	private final double[] mses;
	private double allTimeMse = 0;
	private int mseIndex = 0;
	private int cycle = 0;
	private int allTimeMseCount = 0;

	private final double[][] data;
	private final double[][] answers;
	private final double scaling;
	private final int toDo;

	private final Random rand = new Random();

	public Trainer(double trainingRate, Network net, Fitness trainFit, double percentToDo, int batchSize, int ram) {
		this.trainingRate = trainingRate;
		this.net = net;
		this.ram = ram;
		this.data = trainFit.getData();
		this.answers = trainFit.getAnswers();
		this.toDo = (int) (data.length * percentToDo);
		this.scaling = 2.0 / (toDo * net.param.numOutputs);
		this.mses = new double[ram];
	}

	public void train(int cycles) {
		while (cycle < cycles) {
			if (net instanceof ConvolutionalNetwork) {
				trainStep((ConvolutionalNetwork) net);
			}
			String cycleText = "Cycle " + cycle + "/" + cycles;
			if (ram > 0) {
				cycleText += ", Run Avg Mse = " + Utility.avgString(mses);
			}
			cycleText += ", All Avg Mse = " + Utility.roundString(allTimeMse);
			System.out.println(cycleText);
			cycle++;
		}
	}

	public void trainStep(ConvolutionalNetwork net) {
		net.prepareGrads();

		for (int k = 0; k < toDo; k++) {
			int k1 = this.rand.nextInt(this.data.length);
			double[] x = this.data[k1];
			double[] ans = this.answers[k1];

			double[] eval = net.evaluate(x, k);
			double mse = Utility.mse(eval, ans);

			if (ram > 0 && mses != null) {
				mses[mseIndex] = mse;
				mseIndex = (mseIndex + 1) % ram;
			}
			allTimeMse *= ((double) allTimeMseCount + 1.0) / ((double) allTimeMseCount + 2.0);
			allTimeMse += mse / ((double) allTimeMseCount + 2.0);
			allTimeMseCount++;

			net.computeBackProp(ans, eval, k);
		}

		net.applyGrads(-1 * scaling * trainingRate);
	}
}
