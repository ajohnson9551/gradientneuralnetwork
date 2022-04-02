package core.layer;

import java.io.Serializable;

public abstract class Layer implements Serializable {

	public LayerParameters layerParam;

	public double[][][][] lastX;
	public double[][][][] lastPrime;

	public Layer(LayerParameters layerParam) {
		this.layerParam = layerParam;
	}

	public boolean validateParameters() {
		for (int a : layerParam.inputSize) {
			if (a < 1) {
				return false;
			}
		}
		for (int a : layerParam.outputSize) {
			if (a < 1) {
				return false;
			}
		}
		return true;
	}

	public void setupLasts(int batchSize) {
		lastX = new double[batchSize][][][];
		lastPrime = new double[batchSize][][][];
	}

	public abstract double[][][] evaluate(double[][][] x, int batchIndex);
	public abstract double[][][] getGradientX(int i, int j, int k, int batchIndex);
	public abstract int[][] getGradientXNonzeroRanges(int i, int j, int k);
	public abstract void train(Layer[] grads, double trainingRate);
	public abstract void combineScale(Layer grad, double scale);
	public abstract Layer zeroCopy();
	public abstract void assignGradientInto(Layer receiveGrad, int i, int j, int k, int batchIndex);
}
