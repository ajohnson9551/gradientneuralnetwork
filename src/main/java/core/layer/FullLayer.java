package core.layer;

import core.ActFuncs;
import core.Utility;

import java.util.Arrays;

public class FullLayer extends Layer {

	double[][] A;
	double[] b;

	int[] inputSize;
	int[] outputSize;

	int numInputs;
	int numOutputs;

	public FullLayer(LayerParameters layerParams) {
		super(layerParams);
		assert this.validateParameters();
		this.inputSize = this.layerParam.inputSize;
		this.outputSize = this.layerParam.outputSize;
		int[] iRange = new int[]{0, this.inputSize[0] - 1};
		int[] jRange = new int[]{0, this.inputSize[1] - 1};
		int[] kRange = new int[]{0, this.inputSize[2] - 1};
		this.gradXNonzeroRanges = new int[][]{iRange, jRange, kRange};
		this.numInputs = this.inputSize[0] * this.inputSize[1] * this.inputSize[2];
		this.numOutputs = this.outputSize[0] * this.outputSize[1] * this.outputSize[2];
		this.setupAB(
				numOutputs,
				numInputs,
				true);
	}

	private FullLayer(LayerParameters layerParams, boolean randomize) {
		super(layerParams);
		this.inputSize = this.layerParam.inputSize;
		this.outputSize = this.layerParam.outputSize;
		this.numInputs = this.inputSize[0] * this.inputSize[1] * this.inputSize[2];
		this.numOutputs = this.outputSize[0] * this.outputSize[1] * this.outputSize[2];
		this.setupAB(
				this.numOutputs,
				this.numInputs,
				randomize);
	}

	@Override
	public Layer zeroCopy() {
		return new FullLayer(this.layerParam, false);
	}

	@Override
	public void assignGradientInto(Layer receiveGrad, int i, int j, int k, int batchIndex) {
		for (int ai = 0; ai < this.numOutputs; ai++) {
			for (int aj = 0; aj < this.numInputs; aj++) {
				((FullLayer) receiveGrad).A[ai][aj] = ai == i ? this.lastPrime[batchIndex][i][j][k] * this.lastX[batchIndex][aj % this.inputSize[0]][(aj / this.inputSize[0]) % this.inputSize[1]][aj / (this.inputSize[0] * this.inputSize[1])] : 0;
			}
			((FullLayer) receiveGrad).b[ai] = ai == i ? this.lastPrime[batchIndex][i][j][k] : 0;
		}
	}

	@Override
	public void combineScale(Layer addLayer, double scale) {
		for (int i = 0; i < this.numOutputs; i++) {
			for (int j = 0; j < this.numInputs; j++) {
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
		return (this.layerParam.outputSize[1] == 1 &&
				this.layerParam.outputSize[2] == 1);
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
		double[][][] gradX = new double[this.inputSize[0]][this.inputSize[1]][this.inputSize[2]];
		for (int xk = 0; xk < this.inputSize[2]; xk++) {
			for (int xj = 0; xj < this.inputSize[1]; xj++) {
				for (int xi = 0; xi < this.inputSize[0]; xi++) {
					gradX[xi][xj][xk] = this.lastPrime[batchIndex][i][j][k] * A[i][xi + xj * this.inputSize[0] + xk * this.inputSize[0] * this.inputSize[1]];
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
