package core;

import java.util.Random;

public class BasicNetworkParameters extends NetworkParameters {

	public BasicNetworkParameters(Integer numInputs, Integer numOutputs) {
		super(numInputs, numOutputs);
	}

	public double[][] setupWeights() {
		double[][] weights = initialMatrix(numOutputs, numInputs);
		return weights;
	}

	public double[] setupOffset() {
		double[] offsets = initialColumn(numOutputs);
		return offsets;
	}

	public double[] initialColumn(int rows) {
		double[][] colMat = initialMatrix(rows, 1);
		double[] col = new double[rows];
		for (int i = 0; i < rows; i++) {
			col[i] = colMat[i][0];
		}
		return col;
	}

	public double[][] initialMatrix(int rows, int columns) {
		double[][] out = zeroMatrix(rows, columns);
		for (double[] col : out) {
			for (int i = 0; i < col.length; i++) {
				col[i] = initialValue();
			}
		}
		return out;
	}

	public double initialValue() {
		Random rand = new Random();
		return (2 * rand.nextDouble() - 1);
	}

	public double[][] zeroMatrix(int rows, int columns) {
		return new double[rows][columns];
	}
}
