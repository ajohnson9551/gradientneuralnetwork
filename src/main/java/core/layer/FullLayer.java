package core.layer;

import core.ActFuncs;
import core.Utility;

import java.util.Arrays;

public class FullLayer extends Layer {

	double[][] A;
	double[] b;

	public FullLayer(LayerParameters layerParams) {
		super(layerParams);
		assert this.validateParameters();
		this.setupAB(
				layerParams.outputSize[0] * layerParams.outputSize[1] * layerParams.outputSize[2],
				layerParams.inputSize[0] * layerParams.inputSize[1] * layerParams.inputSize[2],
				true);
		int[] iRange = new int[]{0, this.layerParam.inputSize[0] - 1};
		int[] jRange = new int[]{0, this.layerParam.inputSize[1] - 1};
		int[] kRange = new int[]{0, this.layerParam.inputSize[2] - 1};
		this.gradXNonzeroRanges = new int[][]{iRange, jRange, kRange};
	}

	private FullLayer(LayerParameters layerParams, boolean randomize) {
		super(layerParams);
		this.setupAB(
				layerParams.outputSize[0] * layerParams.outputSize[1] * layerParams.outputSize[2],
				layerParams.inputSize[0] * layerParams.inputSize[1] * layerParams.inputSize[2],
				randomize);
	}

	@Override
	public Layer zeroCopy() {
		return new FullLayer(this.layerParam, false);
	}

	@Override
	public void assignGradientInto(Layer receiveGrad, int i, int j, int k, int batchIndex) {
		for (int ai = 0; ai < this.A.length; ai++) {
			for (int aj = 0; aj < this.A[0].length; aj++) {
				((FullLayer) receiveGrad).A[ai][aj] = ai == i ? this.lastPrime[batchIndex][i][j][k] * this.lastX[batchIndex][aj % this.lastX[batchIndex].length][(aj / this.lastX[batchIndex].length) % this.lastX[batchIndex][0].length][aj / (this.lastX[batchIndex].length * lastX[batchIndex][0].length)] : 0;
			}
			((FullLayer) receiveGrad).b[ai] = ai == i ? this.lastPrime[batchIndex][i][j][k] : 0;
		}
	}

	@Override
	public void combineScale(Layer addLayer, double scale) {
		for (int i = 0; i < A.length; i++) {
			for (int j = 0; j < A[0].length; j++) {
				this.A[i][j] += scale * ((FullLayer) addLayer).A[i][j];
			}
			this.b[i] += scale * ((FullLayer) addLayer).b[i];
		}
	}

	@Override
	public boolean validateParameters() {
		if (!super.validateParameters()) {
			return false;
		}
		return (layerParam.outputSize[1] == 1 &&
				layerParam.outputSize[2] == 1);
	}

	private void setupAB(int rows, int columns, boolean randomize) {
		this.A = new double[rows][columns];
		this.b = new double[rows];
		if (randomize) {
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < columns; j++) {
					this.A[i][j] = Utility.randVal(0, 1);
				}
				this.b[i] = Utility.randVal(0, 1);
			}
		}
	}

	public static double[] convertToArray(double[][][] x) {
		double[] xArr = new double[x.length * x[0].length * x[0][0].length];
		for (int i = 0; i < xArr.length; i++) {
			xArr[i] = x[i % x.length][(i / x.length) % x[0].length][i / (x.length * x[0].length)];
		}
		return xArr;
	}

	public static double[][][] convertToVolume(double[] arr) {
		double[][][] vol = new double[arr.length][1][1];
		for (int i = 0; i < arr.length; i++) {
			vol[i][0][0] = arr[i];
		}
		return vol;
	}

	@Override
	public double[][][] evaluate(double[][][] x, int batchIndex) {
		double[] outArr = Utility.evaluate(this.A, this.b, convertToArray(x), null);
		double[] outCopy = Arrays.copyOf(outArr, outArr.length);

		ActFuncs.getActFuncs().actFuncify(outArr, this.layerParam.actFunc);
		double[][][] out = convertToVolume(outArr);

		ActFuncs.getActFuncs().actFuncPrimeify(outCopy, this.layerParam.actFunc);
		double[][][] save = convertToVolume(outCopy);

		this.lastX[batchIndex] = x;
		this.lastPrime[batchIndex] = save;

		return out;
	}

	@Override
	public double[][][] getGradientX(int i, int j, int k, int batchIndex) {
		double[][][] gradX = new double[this.lastX[batchIndex].length][this.lastX[batchIndex][0].length][this.lastX[batchIndex][0][0].length];
		for (int xk = 0; xk < this.lastX[batchIndex][0][0].length; xk++) {
			for (int xj = 0; xj < this.lastX[batchIndex][0].length; xj++) {
				for (int xi = 0; xi < this.lastX[batchIndex].length; xi++) {
					gradX[xi][xj][xk] = this.lastPrime[batchIndex][i][j][k] * A[i][xi + xj * this.lastX[batchIndex].length + xk * this.lastX[batchIndex].length * this.lastX[batchIndex][0].length];
				}
			}
		}
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
}
