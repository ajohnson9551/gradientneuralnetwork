package core;

import java.util.Random;

public class BasicNetwork implements Network {

	private double[][] weights;
	private double[] offset;

	private final BasicNetworkParameters params;

	public BasicNetwork(BasicNetworkParameters params) {
		this.params = params;
		this.weights = params.setupWeights();
		this.offset = params.setupOffset();
	}

	public double[][] getWeights() {
		return weights;
	}

	public double[] getOffset() {
		return offset;
	}

	public double[] evaluate(double[] x) {
		return Utility.getUtility().evaluate(weights, offset, x, true);
	}

	public double[][] getEmptyGradA() {
		return new double[weights.length][weights[0].length];
	}

	public double[] getEmptyGradB() {
		return new double[offset.length];
	}

	public void gradChange(double[][] gradA, double[] gradB, double trainingRate) {
		for (int i = 0; i < params.numOutputs; i++) {
			for (int j = 0; j < params.numInputs; j++) {
				weights[i][j] = weights[i][j] - (trainingRate * gradA[i][j]);
			}
			offset[i] = offset[i] - (trainingRate * gradB[i]);
		}
	}

	@Override
	public void serialize(String path) {

	}

	@Override
	public Network deserialize(String path) {
		return null;
	}
}
