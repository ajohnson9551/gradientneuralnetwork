package core.network;

import core.layer.Layer;
import core.layer.LayerParameters;

import java.util.Arrays;

public class ConvolutionalNetwork extends Network {

	private final Layer[] layers;
	private Layer[][] grads;
	private final ConvolutionalNetworkParameters param;

	public ConvolutionalNetwork(ConvolutionalNetworkParameters param) {
		super(param);
		this.param = param;
		this.layers = param.layers;
		for (int l = 0; l < this.layers.length; l++) {
			this.layers[l].setupLasts(this.param.batchSize);
			assert l >= this.layers.length - 1 || Arrays.equals(this.layers[l].layerParam.outputSize, this.layers[l + 1].layerParam.inputSize);
		}
	}

	public void prepareGrads() {
		this.grads = new Layer[this.layers.length][param.numOutputs];
		for (int l = this.layers.length - 1; l >= 0; l--) {
			for (int r = 0; r < this.param.numOutputs; r++) {
				this.grads[l][r] = this.layers[l].zeroCopy();
			}
		}
	}

	public void computeBackProp(double[] ans, double[] eval, int batchIndex) {
		double[][][][] gradMult;
		double[][][][] nextGradMult = new double[param.numOutputs][1][1][param.numOutputs];

		for (int i = 0; i < param.numOutputs; i++) {
			nextGradMult[i][0][0][i] = eval[i] - ans[i];
		}

		for (int l = this.layers.length - 1; l >= 0; l--) {
			gradMult = nextGradMult;
			Layer layer = this.layers[l];
			LayerParameters layerParams = layer.layerParam;
			int outputWidth = layerParams.outputSize[0];
			int outputHeight = layerParams.outputSize[1];
			int outputDepth = layerParams.outputSize[2];
			int nextWidth = layerParams.inputSize[0];
			int nextHeight = layerParams.inputSize[1];
			int nextDepth = layerParams.inputSize[2];
			nextGradMult = new double[nextWidth][nextHeight][nextDepth][this.param.numOutputs];
			Layer outputGrad = layer.zeroCopy();
			for (int k = 0; k < outputDepth; k++) {
				for (int j = 0; j < outputHeight; j++) {
					for (int i = 0; i < outputWidth; i++) {
						boolean someNonzero = false;
						for (int r = 0; r < this.param.numOutputs; r++) {
							if (gradMult[i][j][k][r] != 0) {
								someNonzero = true;
								break;
							}
						}
						if (!someNonzero) {
							continue;
						}
						layer.assignGradientInto(outputGrad, i, j, k, batchIndex);
						double[][][] gradX = layer.getGradientX(i, j, k, batchIndex);
						int[][] gradXRanges = layer.getGradientXNonzeroRanges(i, j, k);
						for (int r = 0; r < this.param.numOutputs; r++) {
							Layer grad = layer.zeroCopy();
							if (gradMult[i][j][k][r] != 0) {
								grad.combineScale(outputGrad, gradMult[i][j][k][r]);
								if (l > 0) {
									for (int k1 = gradXRanges[2][0]; k1 <= gradXRanges[2][1]; k1++) {
										for (int j1 = gradXRanges[1][0]; j1 <= gradXRanges[1][1]; j1++) {
											for (int i1 = gradXRanges[0][0]; i1 <= gradXRanges[0][1]; i1++) {
												nextGradMult[i1][j1][k1][r] += gradX[i1][j1][k1] * gradMult[i][j][k][r];
											}
										}
									}
								}
								this.grads[l][r].combineScale(grad, 1);
							}
						}
					}
				}
			}
		}
	}

	public void applyGrads(double trainingRate) {
		for (int batchIndex = 0; batchIndex < this.grads.length; batchIndex++) {
			for (int l = 0; l < this.layers.length; l++) {
				this.layers[l].train(this.grads[l], trainingRate);
			}
		}
	}

	@Override
	public double[] evaluate(double[] x, int batchIndex) {
		double[][][] result = convertToVol(x);
		for (Layer layer : this.layers) {
			result = layer.evaluate(result, batchIndex);
		}
		return convertToArr(result);
	}

	public double[][][] convertToVol(double[] x) {
		int[] inputSize = this.layers[0].layerParam.inputSize;
		double[][][] vol = new double[inputSize[0]][inputSize[1]][inputSize[2]];
		for (int k = 0; k < inputSize[2]; k++) {
			for (int j = 0; j < inputSize[1]; j++) {
				for (int i = 0; i < inputSize[0]; i++) {
					vol[i][j][k] = x[i + j * inputSize[0] + k * inputSize[1] * inputSize[0]];
				}
			}
		}
		return vol;
	}

	public static double[] convertToArr(double[][][] vol) {
		double[] arr = new double[vol.length * vol[0].length * vol[0][0].length];
		for (int i = 0; i < arr.length; i++) {
			arr[i] = vol[i % vol.length][(i / vol.length) % vol[0].length][i / (vol.length * vol[0].length)];
		}
		return arr;
	}
}
