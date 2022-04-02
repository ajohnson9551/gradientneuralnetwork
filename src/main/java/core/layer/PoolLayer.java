package core.layer;

import core.Utility;

import java.util.Arrays;

public class PoolLayer extends Layer {

	int[] coordsOfLargest;

	public PoolLayer(LayerParameters layerParams) {
		super(layerParams);
	}

	@Override
	public double[][][] evaluate(double[][][] x, int batchIndex) {
		double[][][] y = new double[this.layerParam.outputSize[0]][this.layerParam.outputSize[1]][this.layerParam.outputSize[2]];
		int range = this.layerParam.poolSize;
		int stride = this.layerParam.stride;
		for (int k = 0; k < this.layerParam.outputSize[2]; k++) {
			for (int j = 0; j < this.layerParam.outputSize[1]; j++) {
				for (int i = 0; i < this.layerParam.outputSize[0]; i++) {
					y[i][j][k] = maxInRange(x, i * stride, j * stride, k, range);
				}
			}
		}
		this.lastX[batchIndex] = x;
		return y;
	}

	public static double maxInRange(double[][][] x, int i, int j, int k, int range) {
		double max = x[i][j][k];
		for (int i1 = i; i1 < i + range; i1++) {
			for (int j1 = j; j1 < j + range; j1++) {
				double m = Utility.getOrDefault(x, i1, j1, k, -Double.MAX_VALUE);
				if (m > max) {
					max = m;
				}
			}
		}
		return max;
	}

	@Override
	public double[][][] getGradientX(int i, int j, int k, int batchIndex) {
		double[][][] gradX = new double[this.layerParam.inputSize[0]][this.layerParam.inputSize[1]][this.layerParam.inputSize[2]];
		int xi = this.layerParam.stride * i;
		int xj = this.layerParam.stride * j;
		int[] coordsOfLargest = new int[]{xi, xj};
		double max = this.lastX[batchIndex][xi][xj][k];
		for (int xi1 = xi; xi1 < xi + this.layerParam.poolSize; xi1++) {
			for (int xj1 = xj; xj1 < xj + this.layerParam.poolSize; xj1++) {
				double m = Utility.getOrDefault(this.lastX[batchIndex], xi1, xj1, k, -Double.MAX_VALUE);
				if (m > max) {
					max = m;
					coordsOfLargest[0] = xi1;
					coordsOfLargest[1] = xj1;
				}
			}
		}
		this.coordsOfLargest = coordsOfLargest;
		gradX[coordsOfLargest[0]][coordsOfLargest[1]][k] = 1;
		return gradX;
	}

	@Override
	public int[][] getGradientXNonzeroRanges(int i, int j, int k) {
		int[] iRange = new int[]{this.coordsOfLargest[0], this.coordsOfLargest[0]};
		int[] jRange = new int[]{this.coordsOfLargest[1], this.coordsOfLargest[1]};
		int[] kRange = new int[]{k, k};
		this.coordsOfLargest = null;
		return new int[][]{iRange, jRange, kRange};
	}

	@Override
	public void train(Layer[] grads, double trainingRate) {
		// pool does not train
		assert grads instanceof PoolLayer[];
		Arrays.fill(this.lastX, null);
	}

	@Override
	public void combineScale(Layer grad, double scale) {
		// pool does not train
	}

	@Override
	public Layer zeroCopy() {
		// no need to copy, is never changed - return this for maximum efficiency
		return this;
	}

	@Override
	public void assignGradientInto(Layer receiveGrad, int i, int j, int k, int batchIndex) {
		// pool does not train
	}
}
