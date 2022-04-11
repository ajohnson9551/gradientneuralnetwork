package core.layer;

import core.ActFuncs;
import core.Utility;

import java.util.Arrays;

public class ConvolutionalLayer extends Layer {

	double[][][] Cs;

	public ConvolutionalLayer(LayerParameters layerParams) {
		super(layerParams);
		this.setupCs(true);
	}

	private ConvolutionalLayer(LayerParameters layerParams, boolean randomize) {
		super(layerParams);
		this.setupCs(randomize);
	}

	private void setupCs(boolean randomize) {
		this.Cs = new double[this.layerParam.numConvs][2 * this.layerParam.convRadius - 1][2 * this.layerParam.convRadius - 1];
		if (randomize) {
			for (int n = 0; n < this.Cs.length; n++) {
				for (int j = 0; j < this.Cs[0][0].length; j++) {
					 for (int i = 0; i < this.Cs[0].length; i++) {
						 this.Cs[n][i][j] = Utility.randVal(0, 3);
					}
				}
			}
		}
	}

	@Override
	public double[][][] evaluate(double[][][] x, int batchIndex) {
		double[][][] y = new double[this.layerParam.outputSize[0]][this.layerParam.outputSize[1]][this.layerParam.outputSize[2]];
		double[][][] z = new double[this.layerParam.outputSize[0]][this.layerParam.outputSize[1]][this.layerParam.outputSize[2]];
		int convMod = this.layerParam.convMod;
		for (int n = 0; n < this.layerParam.numConvs; n++) {
			for (int k = 0; k < this.layerParam.inputSize[2]; k++) {
				for (int j = 0; j < this.layerParam.outputSize[1]; j++) {
					for (int i = 0; i < this.layerParam.outputSize[0]; i++) {
						double rawConv = convolve(x, i - convMod, j - convMod, k, n);
						y[i][j][n + k * this.layerParam.numConvs] = ActFuncs.getActFuncs().actFunc(rawConv, this.layerParam.actFunc);
						z[i][j][n + k * this.layerParam.numConvs] = ActFuncs.getActFuncs().actFuncPrime(rawConv, this.layerParam.actFunc);
					}
				}
			}
		}
		this.lastX[batchIndex] = x;
		this.lastPrime[batchIndex] = z;
		return y;
	}

	private double convolve(double[][][] x, int i, int j, int k, int n) {
		double result = 0;
		int r = this.layerParam.convRadius - 1;
		for (int cj = -r; cj <= r; cj++) {
			for (int ci = -r; ci <= r; ci++) {
				result += Utility.getOrDefault(x, i + ci, j + cj, k, 0) * Cs[n][ci + r][cj + r];
			}
		}
		return result;
	}

	@Override
	public double[][][] getGradientX(int i, int j, int k, int batchIndex) {
		double[][][] gradX = new double[lastX[batchIndex].length][lastX[batchIndex][0].length][lastX[batchIndex][0][0].length];
		int r = this.layerParam.convRadius - 1;
		int n = k % this.layerParam.numConvs;
		int xk = k / this.layerParam.numConvs;
		int convMod = this.layerParam.convMod;
		for (int cj = -r; cj <= r; cj++) {
			for (int ci = -r; ci <= r; ci++) {
				Utility.setIfCan(gradX, (i - convMod) + ci, (j - convMod) + cj, xk, this.Cs[n][ci + r][cj + r] * this.lastPrime[batchIndex][i][j][k]);
			}
		}
		this.gradXNonzeroRanges[0][0] = Math.max(0, (i - convMod) - r);
		this.gradXNonzeroRanges[0][1] = Math.min(this.layerParam.inputSize[0] - 1, (i - convMod) + r);
		this.gradXNonzeroRanges[1][0] = Math.max(0, (j - convMod) - r);
		this.gradXNonzeroRanges[1][1] = Math.min(this.layerParam.inputSize[1] - 1, (j - convMod) + r);
		this.gradXNonzeroRanges[2][0] = k / this.layerParam.numConvs;
		this.gradXNonzeroRanges[2][1] = k / this.layerParam.numConvs;
		return gradX;
	}

	@Override
	public void train(Layer[] grads, double trainingRate) {
		for (Layer grad : grads) {
			this.combineScale(grad, trainingRate);
		}
		Arrays.fill(this.lastX, null);
		Arrays.fill(this.lastPrime, null);
	}

	@Override
	public void combineScale(Layer grad, double scale) {
		for (int n = 0; n < this.layerParam.numConvs; n++) {
			for (int cj = 0; cj < this.Cs[0][0].length; cj++) {
				for (int ci = 0; ci < this.Cs[0].length; ci++) {
					this.Cs[n][ci][cj] += ((ConvolutionalLayer) grad).Cs[n][ci][cj] * scale;
				}
			}
		}
	}

	@Override
	public Layer zeroCopy() {
		return new ConvolutionalLayer(this.layerParam, false);
	}

	@Override
	public void assignGradientInto(Layer receiveGrad, int i, int j, int k, int batchIndex) {
		receiveGrad = receiveGrad.zeroCopy(); // unlike full, needs to reset receiveGrad
		int r = this.layerParam.convRadius - 1;
		int nc = this.layerParam.numConvs;
		int convMod = this.layerParam.convMod;
		for (int cj = -r; cj <= r; cj++) {
			for (int ci = -r; ci <= r; ci++) {
				((ConvolutionalLayer) receiveGrad).Cs[k % nc][ci + r][cj + r] = Utility.getOrDefault(this.lastX[batchIndex], (i - convMod) + ci, (j - convMod) + cj, k / nc, 0) * this.lastPrime[batchIndex][i][j][k];
			}
		}
	}
}
