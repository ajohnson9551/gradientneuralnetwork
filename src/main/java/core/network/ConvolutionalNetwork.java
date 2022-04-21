package core.network;

import com.aparapi.Kernel;
import com.aparapi.Range;
import core.layer.Layer;
import core.layer.LayerParameters;

import java.util.Arrays;

public class ConvolutionalNetwork extends Network {

	private final Layer[] layers;
	private final ConvolutionalNetworkParameters param;

	private transient Layer[][] grads;

	private transient int batchIndex;

	private final transient double[][][][][] gradMults;

	private transient final int[] outputWidths;
	private transient final int[] outputHeights;
	private transient final int[] outputDepths;

	private transient final Kernel[] kernels;
	private transient final Range[] ranges;

	public ConvolutionalNetwork(ConvolutionalNetworkParameters param) {
		super(param);
		this.param = param;
		this.layers = param.layers;

		gradMults = new double[this.layers.length][][][][];

		kernels = new Kernel[this.layers.length];
		ranges = new Range[this.layers.length];

		outputWidths = new int[this.layers.length];
		outputHeights = new int[this.layers.length];
		outputDepths = new int[this.layers.length];

		for (int l = 0; l < this.layers.length; l++) {
			this.layers[l].setupLasts(this.param.batchSize);
			assert l >= this.layers.length - 1 || Arrays.equals(this.layers[l].layerParam.outputSize, this.layers[l + 1].layerParam.inputSize);

			this.outputWidths[l] = this.layers[l].layerParam.outputSize[0];
			this.outputHeights[l] = this.layers[l].layerParam.outputSize[1];
			this.outputDepths[l] = this.layers[l].layerParam.outputSize[2];

			ranges[l] = Range.create(outputWidths[l] * outputHeights[l] * outputDepths[l]);

			int finalL = l;
			kernels[l] = new Kernel() {
				@Override
				public void run() {
					int ijk = getGlobalId();
					int i = ijk % outputWidths[finalL];
					int j = (ijk / outputWidths[finalL]) % outputHeights[finalL];
					int k = (ijk / (outputHeights[finalL] * outputWidths[finalL])) % outputDepths[finalL];
					Layer layer = layers[finalL];
					LayerParameters layerParams = layer.layerParam;
					int nextWidth = layerParams.inputSize[0];
					int nextHeight = layerParams.inputSize[1];
					int nextDepth = layerParams.inputSize[2];
					if (finalL > 0) {
						gradMults[finalL - 1] = new double[nextWidth][nextHeight][nextDepth][param.numOutputs];
					}
					Layer outputGrad = layer.zeroCopy();
					int lowestNonzero = param.numOutputs;
					for (int r = 0; r < param.numOutputs; r++) {
						if (gradMults[finalL][i][j][k][r] != 0) {
							lowestNonzero = r;
							break;
						}
					}
					if (lowestNonzero == param.numOutputs) {
						return;
					}
					layers[finalL].assignGradientInto(outputGrad, i, j, k, batchIndex);
					double[][][] gradX = layers[finalL].getGradientX(i, j, k, batchIndex);
					int[][] gradXRanges = layers[finalL].gradXNonzeroRanges;
					for (int r = lowestNonzero; r < param.numOutputs; r++) {
						if (gradMults[finalL][i][j][k][r] != 0) {
							if (finalL > 0) {
								for (int k1 = gradXRanges[2][0]; k1 <= gradXRanges[2][1]; k1++) {
									for (int j1 = gradXRanges[1][0]; j1 <= gradXRanges[1][1]; j1++) {
										for (int i1 = gradXRanges[0][0]; i1 <= gradXRanges[0][1]; i1++) {
											gradMults[finalL - 1][i1][j1][k1][r] += gradX[i1][j1][k1] * gradMults[finalL][i][j][k][r];
										}
									}
								}
							}
							grads[finalL][r].combineScale(outputGrad, gradMults[finalL][i][j][k][r]);
						}
					}
				}
			};
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
		this.batchIndex = batchIndex;

		gradMults[this.layers.length - 1] = new double[this.param.numOutputs][1][1][this.param.numOutputs];
		for (int i = 0; i < this.param.numOutputs; i++) {
			gradMults[this.layers.length - 1][i][0][0][i] = eval[i] - ans[i];
		}

		for (int l = this.layers.length - 1; l >= 0; l--) {
			this.kernels[l].execute(this.ranges[l]);
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
