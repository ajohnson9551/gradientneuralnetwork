package core;

import java.io.*;
import java.util.Arrays;

public class LayeredNetwork implements Network {

	private final LayeredNetworkParameters params;
	private final double[][][] weights;
	private final double[][] offsets;

	private final transient Utility util = Utility.getUtility();

	public LayeredNetwork(String path) {
		LayeredNetwork net = (LayeredNetwork) this.deserialize(path);
		this.params = net.params;
		this.weights = net.weights;
		this.offsets = net.offsets;
	}

	public LayeredNetwork(LayeredNetworkParameters params) {
		this.params = params;
		this.weights = params.setupWeights(false);
		this.offsets = params.setupOffsets(false);
	}

	@Override
	public double[] evaluate(double[] x) {
		return evaluate(x, weights.length);
	}

	public double[] evaluate(double[] x, int toLayer) {
		if (toLayer == 0) {
			return Arrays.copyOf(x, x.length);
		}
		double[] result;
		result = util.evaluate(weights[0], offsets[0], x, true);
		for (int i = 1; i < Math.min(toLayer, weights.length); i++) {
			result = util.evaluate(weights[i], offsets[i], result, true);
		}
		return result;
	}

	public void gradChange(double[][][] gradAs, double[][] gradBs, double trainingRate) {
		for (int v = 0; v < gradAs.length; v++) {
			for (int i = 0; i < gradAs[v].length; i++) {
				for (int j = 0; j < gradAs[v][0].length; j++) {
					weights[v][i][j] = weights[v][i][j] - (trainingRate * gradAs[v][i][j]);
				}
				offsets[v][i] = offsets[v][i]- (trainingRate * gradBs[v][i]);
			}
		}
	}

	public double[][][] getWeights() {
		return weights;
	}

	public double[][] getOffsets() {
		return offsets;
	}

	public int getNumHiddenLayers() {
		return params.hiddenLayerSizes.length;
	}

	public double[][][] getEmptyGradAs() {
		return params.setupWeights(true);
	}

	public double[][] getEmptyGradBs() {
		return params.setupOffsets(true);
	}

	@Override
	public void serialize(String path) {
		try {
			FileOutputStream fileOut = new FileOutputStream(path + "/network.ser");
			ObjectOutputStream out = new ObjectOutputStream(fileOut);
			out.writeObject(this);
			out.close();
			fileOut.close();
		} catch (IOException i) {
			i.printStackTrace();
		}
	}

	@Override
	public Network deserialize(String path) {
		LayeredNetwork net;
		try {
			FileInputStream fileIn = new FileInputStream(path + "/network.ser");
			ObjectInputStream in = new ObjectInputStream(fileIn);
			net = (LayeredNetwork) in.readObject();
			in.close();
			fileIn.close();
		} catch (IOException | ClassNotFoundException e) {
			e.printStackTrace();
			return null;
		}
		return net;
	}
}
